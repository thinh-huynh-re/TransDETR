# ------------------------------------------------------------------------
# Copyright (c) 2021 Zhejiang University-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
DETR model and criterion classes.
"""
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import List
import math
from util import box_ops
from util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    is_dist_avail_and_initialized,
    inverse_sigmoid,
)

from models.structures import (
    Instances,
    Boxes,
    matched_boxlist_iou,
)
from .structures.conv_bn_relu import Conv_BN_ReLU
from detectron2.layers.roi_align_rotated import ROIAlignRotated
from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer_plus import build_deforamble_transformer
from .qim import build as build_query_interaction_layer
from .memory_bank import build_memory_bank
from .deformable_detr import SetCriterion, MLP
from .segmentation import sigmoid_focal_loss
from .head.rec_head_ctc import PAN_PP_RecHead_CTC
from datasets.data_tools import get_vocabulary
import time


class ClipMatcher(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, losses, language="LOWERCASE"):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0
        # CHINESE  LOWERCASE
        self.voc, self.char2id, self.id2char = get_vocabulary(language, use_ctc=True)
        self.blank = self.char2id["PAD"]

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        frame_id = self._current_frame_idx - 1
        gt_instances = self.gt_instances[frame_id]
        outputs = {
            "pred_logits": track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = (
            track_instances.matched_gt_idxes
        )  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss(
            "labels",
            outputs=outputs,
            gt_instances=[gt_instances],
            indices=[(src_idx, tgt_idx)],
            num_boxes=1,
        )
        self.losses_dict.update(
            {
                "frame_{}_track_{}".format(frame_id, key): value
                for key, value in track_losses.items()
            }
        )

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(
            num_samples, dtype=torch.float, device=self.sample_device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "rotate": self.loss_rotate,
            "rec": self.loss_rec,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(
        self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes
    ):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        # TODO: Make this filter process more elegant.  rec_masks = gt_instances.texts_ignored
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)],
            dim=0,
        )

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat(
            [
                gt_per_img.obj_ids[i]
                for gt_per_img, (_, i) in zip(gt_instances, indices)
            ],
            dim=0,
        )  # size(16)

        # for ignored text, don't calculate regression loss
        target_text_ids = torch.cat(
            [
                gt_per_img.texts_ignored[i]
                for gt_per_img, (_, i) in zip(gt_instances, indices)
            ],
            dim=0,
        )  # size(16)

        mask = (target_obj_ids != -1) * (target_text_ids == 1)
        #         mask = target_obj_ids != -1

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction="none")
        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
                box_ops.box_cxcywh_to_xyxy(target_boxes[mask]),
            )
        )

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def loss_labels(
        self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False
    ):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        ignored_classes = torch.full(
            src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device
        )

        # The matched gt for disappear track query is set -1.
        labels = []
        ignored_label = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J)
            labels_ignored_per_img = torch.ones_like(J)
            # set labels of track-appear slots to 0.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
                labels_ignored_per_img[J != -1] = gt_per_img.texts_ignored[J[J != -1]]
            labels.append(labels_per_img)
            ignored_label.append(labels_ignored_per_img)

        target_classes_o = torch.cat(labels)
        target_classes_o_ignored = torch.cat(ignored_label)
        target_classes[idx] = target_classes_o
        ignored_classes[idx] = target_classes_o_ignored

        if self.focal_loss:
            gt_labels_target = F.one_hot(
                target_classes, num_classes=self.num_classes + 1
            )[
                :, :, :-1
            ]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = sigmoid_focal_loss(
                src_logits.flatten(1) * ignored_classes,
                gt_labels_target.flatten(1) * ignored_classes,
                alpha=0.25,
                gamma=2,
                num_boxes=num_boxes,
                mean_in_dim1=False,
            )
            loss_ce = loss_ce.sum()
        else:
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_rec(
        self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False
    ):
        assert "pre_rec" in outputs
        # 过滤掉rec mask的结果
        gt_instances = gt_instances[0]
        out_recognition = outputs["pre_rec"]
        rec_masks = gt_instances.texts_ignored

        if torch.sum(rec_masks) == 0:
            losses = {
                "loss_rec": out_recognition.new_full((1,), 0.0, dtype=torch.float32)[0]
            }
            return losses

        preds = out_recognition[rec_masks == 1, :, :]
        targets = gt_instances.word[rec_masks == 1, :]

        preds = preds.permute(1, 0, 2)  # 32 N 4714
        target_lengths = (targets != self.blank).long().sum(dim=-1)

        trimmed_targets = [
            t[:l] for t, l in zip(targets, target_lengths)
        ]  # 这里取出了制作label时所有的结果
        targets = torch.cat(trimmed_targets)
        x = F.log_softmax(preds, dim=-1)
        input_lengths = torch.full((x.size(1),), x.size(0), dtype=torch.long)

        loss_rec = F.ctc_loss(
            x,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank,
            zero_infinity=True,
        )
        if loss_rec.view(-1)[0] < 0.002:
            losses = {
                "loss_rec": out_recognition.new_full((1,), 0.0, dtype=torch.float32)[0]
            }
            return losses

        losses = {"loss_rec": loss_rec.view(-1)[0]}

        return losses

    def loss_rotate(
        self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False
    ):
        """Classification loss (NLL)
        gt_instances dicts must contain the key "pred_rotate" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_rotate" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs["pred_rotate"]

        target_rotate = torch.full(
            src_logits.shape[:2], 0.0, dtype=torch.float, device=src_logits.device
        )

        ignored_classes = torch.full(
            src_logits.shape[:2], 1.0, dtype=torch.long, device=src_logits.device
        )

        # The matched gt for disappear track query is set -1.
        labels = []
        ignored_label = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            rotate_per_img = torch.zeros_like(J, dtype=torch.float)
            labels_ignored_per_img = torch.ones_like(J, dtype=torch.long)
            # set labels of track-appear slots to 0.

            if len(gt_per_img) > 0:
                rotate_per_img[J != -1] = gt_per_img.rotate[J[J != -1]]
                labels_ignored_per_img[J != -1] = gt_per_img.texts_ignored[J[J != -1]]

            labels.append(rotate_per_img)
            ignored_label.append(labels_ignored_per_img)

        target_rotate_o = torch.cat(labels)
        target_rotate[idx] = target_rotate_o

        target_classes_o_ignored = torch.cat(ignored_label)
        ignored_classes[idx] = target_classes_o_ignored
        ignored_classes = ignored_classes.float()
        #         target_rotate_o = torch.cat([t["rotate"][J] for t, (_, J) in zip(targets, indices)])
        #         target_rotate = torch.full(pred_rotate.shape[:2], 0.0,
        #                                     dtype=torch.float, device=pred_rotate.device)
        #         target_rotate[idx] = target_rotate_o
        ignored = torch.full(
            src_logits.shape[:2], 0.0, dtype=torch.float, device=src_logits.device
        )
        ignored[idx] = 1.0

        pred_rotate = (src_logits.sigmoid() - 0.5) * math.pi
        angle_loss = 1 - torch.cos(
            pred_rotate * ignored.unsqueeze(-1) * ignored_classes.unsqueeze(-1)
            - target_rotate.unsqueeze(-1) * ignored_classes.unsqueeze(-1)
        )
        sum_ = torch.clamp(ignored.sum(), 1, 10000)
        losses = {"loss_angle": angle_loss.sum() / (sum_ * num_boxes)}

        return losses

    def match_for_single_frame(self, outputs: dict, pred_rec):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        gt_instances_i = self.gt_instances[
            self._current_frame_idx
        ]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux["track_instances"]
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_rotate_i = track_instances.pred_rotate  # predicted angle of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.
        pred_rec_i = pred_rec

        obj_idxes = gt_instances_i.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {
            obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)
        }
        outputs_i = {
            "pred_logits": pred_logits_i.unsqueeze(0),
            "pred_boxes": pred_boxes_i.unsqueeze(0),
            "pred_rotate": pred_rotate_i.unsqueeze(0),
            "pre_rec": pred_rec_i,
        }

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        for j in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[j].item()
            # set new target idx.
            if obj_id >= 0:
                if obj_id in obj_idx_to_gt_idx:
                    track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[obj_id]
                else:
                    num_disappear_track += 1
                    track_instances.matched_gt_idxes[j] = -1  # track-disappear case.
            else:
                track_instances.matched_gt_idxes[j] = -1

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(
            pred_logits_i.device
        )
        matched_track_idxes = track_instances.obj_idxes >= 0  # occu
        prev_matched_indices = torch.stack(
            [
                full_track_idxes[matched_track_idxes],
                track_instances.matched_gt_idxes[matched_track_idxes],
            ],
            dim=1,
        ).to(pred_logits_i.device)

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances_i)).to(pred_logits_i.device)
        tgt_state[tgt_indexes] = 1
        untracked_tgt_indexes = torch.arange(len(gt_instances_i)).to(
            pred_logits_i.device
        )[tgt_state == 0]
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]

        def match_for_single_decoder_layer(unmatched_outputs, matcher):
            new_track_indices = matcher(
                unmatched_outputs, [untracked_gt_instances]
            )  # list[tuple(src_idx, tgt_idx)]

            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt.
            new_matched_indices = torch.stack(
                [unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]], dim=1
            ).to(pred_logits_i.device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            "pred_logits": track_instances.pred_logits[unmatched_track_idxes].unsqueeze(
                0
            ),
            "pred_boxes": track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(
                0
            ),
            "pred_rotate": track_instances.pred_rotate[unmatched_track_idxes].unsqueeze(
                0
            ),
        }
        new_matched_indices = match_for_single_decoder_layer(
            unmatched_outputs, self.matcher
        )

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[
            new_matched_indices[:, 1]
        ].long()
        track_instances.matched_gt_idxes[
            new_matched_indices[:, 0]
        ] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (
            track_instances.matched_gt_idxes >= 0
        )
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        active_track_angle = track_instances.pred_rotate[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[
                track_instances.matched_gt_idxes[active_idxes]
            ]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(
                Boxes(active_track_boxes), Boxes(gt_boxes)
            )

            gt_angle = gt_instances_i.rotate[
                track_instances.matched_gt_idxes[active_idxes]
            ]
            active_track_angle = (active_track_angle.sigmoid() - 0.5) * math.pi

            track_instances.angle[active_idxes] = torch.abs(
                active_track_angle[0] - gt_angle
            )

        #             track_instances.iou[active_idxes] = matched_boxlist_rotated_iou(active_track_boxes, gt_boxes, active_track_angle, gt_angle)

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            new_track_loss = self.get_loss(
                loss,
                outputs=outputs_i,
                gt_instances=[gt_instances_i],
                indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                num_boxes=1,
            )
            self.losses_dict.update(
                {
                    "frame_{}_{}".format(self._current_frame_idx, key): value
                    for key, value in new_track_loss.items()
                }
            )

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                unmatched_outputs_layer = {
                    "pred_logits": aux_outputs["pred_logits"][
                        0, unmatched_track_idxes
                    ].unsqueeze(0),
                    "pred_boxes": aux_outputs["pred_boxes"][
                        0, unmatched_track_idxes
                    ].unsqueeze(0),
                    "pred_rotate": aux_outputs["pred_rotate"][
                        0, unmatched_track_idxes
                    ].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(
                    unmatched_outputs_layer, self.matcher
                )
                matched_indices_layer = torch.cat(
                    [new_matched_indices_layer, prev_matched_indices], dim=0
                )
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if loss == "rec":
                        continue
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        gt_instances=[gt_instances_i],
                        indices=[
                            (matched_indices_layer[:, 0], matched_indices_layer[:, 1])
                        ],
                        num_boxes=1,
                    )
                    self.losses_dict.update(
                        {
                            "frame_{}_aux{}_{}".format(
                                self._current_frame_idx, i, key
                            ): value
                            for key, value in l_dict.items()
                        }
                    )
        self._step()
        return track_instances

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.5, filter_score_thresh=0.2, miss_tolerance=1):
        # dataset score_thresh, filter_score_thresh
        # ICDAR15 0.5 0.2  3(COCOText: 0.6 0.4)
        # YVT 0.3 0.2
        # Minetto 0.4 0.3
        # ICDAR13(detection) 0.4 0.25
        # BOVText 0.74. 0.57   3
        # Pretrain on SynthText, test on ICDAR15   0.8  0.6
        # Pretrain on VISD, test on ICDAR15   0.9  0.7  (finetune 0.7 0.4)
        # Pretrain on UnrealText, test on ICDAR15   0.6  0.4
        # Pretrain on Video SynthText, test on ICDAR15   0.6  0.4
        # Pretrain on Video SynthText(activity), test on ICDAR15   0.6  0.4
        # Pretrain on Video SynthText(image activity), test on ICDAR15   0.8  0.6
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def filter_dt_by_score(self, dt_instances: Instances) -> Instances:
        keep = dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    def update(self, track_instances: Instances):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0

        #         filter_ins = self.filter_dt_by_score(track_instances)

        for i in range(len(track_instances)):
            if (
                track_instances.obj_idxes[i] == -1
                and track_instances.scores[i] >= self.score_thresh
            ):
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1

            elif (
                track_instances.obj_idxes[i] >= 0
                and track_instances.scores[i] < self.filter_score_thresh
            ):
                track_instances.obj_idxes[i] = -1


#                 track_instances.disappear_time[i] += 1
#                 if track_instances.disappear_time[i] >= self.miss_tolerance:
# #                     Set the obj_id to -1.
# #                     Then this track will be removed by TrackEmbeddingLayer.
#                     track_instances.obj_idxes[i] = -1


class TrackerPostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes
        out_rotate = track_instances.pred_rotate
        out_rec = track_instances.pred_rec
        rotate = (out_rotate.sigmoid() - 0.5) * math.pi

        prob = out_logits.sigmoid()
        # prob = out_logits[...,:1].sigmoid()
        scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        # rec
        rec_probs = F.softmax(out_rec, dim=2)
        preds_max_prob, out_rec_decoded = rec_probs.max(dim=-1)  # N 32

        track_instances.word = out_rec_decoded
        track_instances.word_max_prob = preds_max_prob
        track_instances.rotate = rotate
        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = labels
        track_instances.roi = track_instances.roi_feature
        track_instances.remove("pred_logits")
        track_instances.remove("pred_boxes")
        track_instances.remove("pred_rotate")
        track_instances.remove("pred_rec")
        track_instances.remove("roi_feature")

        return track_instances


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransDETR(nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        num_feature_levels,
        criterion,
        track_embed,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        memory_bank=None,
        charater=38,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        # English:38      Chinese+English: 4713
        self.bilingual = True
        self.character = charater
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.rotate_embed = nn.Linear(hidden_dim, 1)

        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        # Mask RoI+ Recognition
        self.reduce_layer4 = Conv_BN_ReLU(256, 128)
        self.reduce_layer3 = Conv_BN_ReLU(256, 128)
        self.reduce_layer2 = Conv_BN_ReLU(256, 128)
        self.reduce_layer1 = Conv_BN_ReLU(256, 128)

        self.conv = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.roirotate = ROIAlignRotated((8, 32), spatial_scale=(1.0), sampling_ratio=0)
        self.rec_head = PAN_PP_RecHead_CTC(self.character)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.rotate_embed.bias.data = torch.ones(num_classes) * bias_value

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
            (transformer.decoder.num_layers + 1)
            if two_stage
            else transformer.decoder.num_layers
        )
        if with_box_refine:
            self.rotate_embed = _get_clones(self.rotate_embed, num_pred)
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.rotate_embed = nn.ModuleList(
                [self.rotate_embed for _ in range(num_pred)]
            )
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase()
        self.criterion = criterion
        self.memory_bank = memory_bank
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embed.weight.shape  # (300, 512)
        device = self.query_embed.weight.device
        track_instances.ref_pts = self.transformer.reference_points(
            self.query_embed.weight[:, : dim // 2]
        )
        track_instances.query_pos = self.query_embed.weight
        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device
        )
        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long, device=device
        )
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.angle = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.rec = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        #         track_instances.roi = torch.zeros((len(track_instances),), dtype=torch.float, device=device)

        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 4), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )
        track_instances.pred_rotate = torch.zeros(
            (len(track_instances), 1), dtype=torch.float, device=device
        )
        # English:38      Chinese+English: 4713
        track_instances.pred_rec = torch.zeros(
            (len(track_instances), 32, self.character), dtype=torch.float, device=device
        )
        track_instances.roi_feature = torch.zeros(
            (len(track_instances), 128, 8, 32), dtype=torch.float, device=device
        )

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32,
            device=device,
        )
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len), dtype=torch.bool, device=device
        )
        track_instances.save_period = torch.zeros(
            (len(track_instances),), dtype=torch.float32, device=device
        )

        return track_instances.to(self.query_embed.weight.device)

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_rotate):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        #         # as a dict having both a Tensor and a list.
        #         return [{'pred_logits': a, 'pred_boxes': b, }
        #                 for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_rotate": c}
            for a, b, c in zip(
                outputs_class[:-1], outputs_coord[:-1], outputs_rotate[:-1]
            )
        ]

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode="bilinear")

    def rec_upsample(self, x, output_size):
        return F.upsample(x, size=output_size, mode="bilinear")

    def _forward_single_image(
        self,
        samples,
        track_instances: Instances,
        taget=None,
        h=None,
        w=None,
        time_cost=None,
    ):
        if not self.training and time_cost != None:
            torch.cuda.synchronize()
            start = time.time()

        features, pos, layer1 = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        if not self.training and time_cost != None:
            torch.cuda.synchronize()
            time_cost.update(dict(backbone_time=time.time() - start))
            start = time.time()

        f1, mask = layer1.decompose()
        features_rec = [f1]

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            temp_src = self.input_proj[l](src)
            srcs.append(temp_src)
            features_rec.append(temp_src)
            masks.append(mask)
            assert mask is not None

        if not self.training and time_cost != None:
            torch.cuda.synchronize()
            time_cost.update(dict(nect_time=time.time() - start))
            start = time.time()

        # recognition
        f1 = self.reduce_layer1(features_rec[0])
        f2 = self.reduce_layer2(features_rec[1])
        f3 = self.reduce_layer3(features_rec[2])
        f4 = self.reduce_layer4(features_rec[3])

        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        rec_feature = self.conv(f)
        rec_feature = self.relu(self.bn(rec_feature))
        rec_feature = self.rec_upsample(rec_feature, (h, w))  # 上采样到原始输入特征大小

        if not self.training and time_cost != None:
            torch.cuda.synchronize()
            time_cost.update(dict(upsample_time=time.time() - start))
            start = time.time()

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(
            srcs, masks, pos, track_instances.query_pos, ref_pts=track_instances.ref_pts
        )

        if not self.training and time_cost != None:
            torch.cuda.synchronize()
            time_cost.update(dict(transformer_time=time.time() - start))
            start = time.time()

        outputs_classes = []
        outputs_coords = []
        outputs_rotates = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_rotate = self.rotate_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_rotates.append(outputs_rotate)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_rotate = torch.stack(outputs_rotates)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        ref_pts_all = torch.cat(
            [init_reference[None], inter_references[:, :, :, :2]], dim=0
        )
        #         print(ref_pts_all.shape)
        #         out = {'pred_logits': outputs_class[-1], 'pred_rotate': outputs_rotate[-1], 'pred_boxes': outputs_coord[-1], 'ref_pts': ref_pts_all[5]}
        out = {
            "pred_logits": outputs_class[-1],
            "pred_rotate": outputs_rotate[-1],
            "pred_boxes": outputs_coord[-1],
            "ref_pts": ref_pts_all[2],
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_rotate
            )

        with torch.no_grad():
            if self.training:
                track_scores = outputs_class[-1, 0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = outputs_class[-1, 0, :, 0].sigmoid()

        track_instances.scores = track_scores
        track_instances.pred_logits = outputs_class[-1, 0]
        track_instances.pred_boxes = outputs_coord[-1, 0]
        track_instances.pred_rotate = outputs_rotate[-1, 0]
        track_instances.output_embedding = hs[-1, 0]

        if self.training:
            # rotated roi
            rois = torch.full(
                (taget.boxes.shape[0], 6),
                0.0,
                dtype=torch.float,
                device=rec_feature.device,
            )
            cwh = taget.boxes * torch.tensor(
                [w, h, w, h], dtype=torch.float, device=rec_feature.device
            )
            angle = taget.rotate / math.pi * 180
            rois[:, 1:5] = cwh
            rois[:, 5] = angle
            roi_features = self.roirotate(rec_feature, rois)
            out_rec = self.rec_head(roi_features)  # N * 32 (最大字符串长度) * voc_size 这是识别的信息

            #             out.update({'out_rec':out_rec})
            #             track_instances.pred_rec = out_rec

            # the track id will be assigned by the mather.
            out["track_instances"] = track_instances
            track_instances = self.criterion.match_for_single_frame(out, out_rec)
        else:
            #             keep = track_instances.scores > self.track_base.filter_score_thresh
            #             track_instances = track_instances[keep]

            # each track will be assigned an unique global id by the track base.
            self.track_base.update(track_instances)

            if not self.training and time_cost != None:
                torch.cuda.synchronize()
                time_cost.update(dict(det_head_time=time.time() - start))
                start = time.time()
            #             recog_indx = track_instances.scores>0.5
            keep = track_instances.obj_idxes >= 0
            track_instances = track_instances[keep]

            if track_instances.pred_boxes.shape[0] != 0:
                rois = torch.full(
                    (track_instances.pred_boxes.shape[0], 6),
                    0.0,
                    dtype=torch.float,
                    device=rec_feature.device,
                )
                cwh = track_instances.pred_boxes * torch.tensor(
                    [w, h, w, h], dtype=torch.float, device=rec_feature.device
                )
                angle = track_instances.pred_rotate / math.pi * 180
                rois[:, 1:5] = cwh
                rois[:, 5:6] = angle
                roi_features = self.roirotate(rec_feature, rois)
                out_rec = self.rec_head(
                    roi_features
                )  # N * 32 (最大字符串长度) * voc_size 这是识别的信息
                track_instances.pred_rec = out_rec
                track_instances.roi_feature = roi_features

            else:
                # English:38      Chinese+English: 4713
                out_rec = torch.zeros(
                    (len(track_instances), 32, self.character),
                    dtype=torch.float,
                    device=track_instances.pred_boxes.device,
                )
                roi = torch.zeros(
                    (len(track_instances), 128, 8, 32),
                    dtype=torch.float,
                    device=track_instances.pred_boxes.device,
                )
                track_instances.pred_rec = out_rec
                track_instances.roi_feature = roi

        if not self.training and time_cost != None:
            torch.cuda.synchronize()
            time_cost.update(dict(rec_head_time=time.time() - start))
            start = time.time()

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
            # track_instances.track_scores = track_instances.track_scores[..., 0]
            # track_instances.scores = track_instances.track_scores.sigmoid()
            if self.training:
                self.criterion.calc_loss_for_track_scores(track_instances)
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.track_embed(tmp)
        out["track_instances"] = out_track_instances

        if not self.training and time_cost != None:
            torch.cuda.synchronize()
            time_cost.update(dict(memory_embed_time=time.time() - start))
            start = time.time()

        return out, time_cost

    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks()

        time_cost = {}
        res, time_cost = self._forward_single_image(
            img,
            track_instances=track_instances,
            h=ori_img_size[0],
            w=ori_img_size[1],
            time_cost=time_cost,
        )

        track_instances = res["track_instances"]

        if not self.training and time_cost != None:
            torch.cuda.synchronize()
            start = time.time()

        track_instances = self.post_process(track_instances, ori_img_size)
        ret = {"track_instances": track_instances}

        if not self.training and time_cost != None:
            torch.cuda.synchronize()
            time_cost.update(dict(postprocess_time=time.time() - start))
            start = time.time()

        if "ref_pts" in res:
            ref_pts = res["ref_pts"]
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret["ref_pts"] = ref_pts
        return ret, time_cost

    def forward(self, data: dict):
        if self.training:
            self.criterion.initialize_for_single_clip(data["gt_instances"])
        frames = data["imgs"]  # list of Tensor.
        gt_instances = data["gt_instances"]  # list of label.
        outputs = {
            "pred_logits": [],
            "pred_boxes": [],
            "pred_rotate": [],
        }

        track_instances = self._generate_empty_tracks()
        for frame, taget in zip(frames, gt_instances):
            h, w = frame.shape[-2:]
            if not isinstance(frame, NestedTensor):
                frame = nested_tensor_from_tensor_list([frame])

            frame_res, time_cost = self._forward_single_image(
                frame, track_instances, taget, h, w
            )
            track_instances = frame_res["track_instances"]
            outputs["pred_logits"].append(frame_res["pred_logits"])
            outputs["pred_boxes"].append(frame_res["pred_boxes"])
            outputs["pred_rotate"].append(frame_res["pred_rotate"])

        if not self.training:
            outputs["track_instances"] = track_instances
        else:
            outputs["losses_dict"] = self.criterion.losses_dict
        return outputs


def build(args):
    dataset_to_num_classes = {
        "coco": 91,
        "coco_panoptic": 250,
        "e2e_mot": 1,
        "e2e_joint": 1,
        "e2e_static_mot": 1,
        "Text": 1,
        "VideoText": 1,
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    query_interaction_layer = build_query_interaction_layer(
        args, args.query_interaction_layer, d_model, hidden_dim, d_model * 2
    )

    img_matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        # if not args.only_rec:
        #     weight_dict.update(
        #         {
        #             "frame_{}_loss_ce".format(i): args.cls_loss_coef,
        #             "frame_{}_loss_bbox".format(i): args.bbox_loss_coef,
        #             "frame_{}_loss_giou".format(i): args.giou_loss_coef,
        #             "frame_{}_loss_angle".format(i): 50.0,
        #             "frame_{}_loss_rec".format(i): 5,
        #         }
        #     )
        # else:
            # BOVText
        weight_dict.update(
            {
                "frame_{}_loss_ce".format(i): 0,
                "frame_{}_loss_bbox".format(i): 0,
                "frame_{}_loss_giou".format(i): 0,
                "frame_{}_loss_angle".format(i): 0,
                "frame_{}_loss_rec".format(i): 1,
            }
        )

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                # if not args.only_rec:
                #     weight_dict.update(
                #         {
                #             "frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                #             "frame_{}_aux{}_loss_bbox".format(
                #                 i, j
                #             ): args.bbox_loss_coef,
                #             "frame_{}_aux{}_loss_giou".format(
                #                 i, j
                #             ): args.giou_loss_coef,
                #             "frame_{}_aux{}_loss_angle".format(i, j): 50.0,
                #             "frame_{}_aux{}_loss_rec".format(i, j): 5,
                #         }
                #     )
                # else:
                    # BOVText
                weight_dict.update(
                    {
                        "frame_{}_aux{}_loss_ce".format(i, j): 0,
                        "frame_{}_aux{}_loss_bbox".format(i, j): 0,
                        "frame_{}_aux{}_loss_giou".format(i, j): 0,
                        "frame_{}_aux{}_loss_angle".format(i, j): 0,
                        "frame_{}_aux{}_loss_rec".format(i, j): 1,
                    }
                )

    if args.memory_bank_type is not None and len(args.memory_bank_type) > 0:
        memory_bank = build_memory_bank(args, d_model, hidden_dim, d_model * 2)
        #         for i in range(num_frames_per_batch):
        #             weight_dict.update({"frame_{}_track_loss_ce".format(i): args.cls_loss_coef})
        for i in range(num_frames_per_batch):
            weight_dict.update({"frame_{}_track_loss_ce".format(i): 0})
    else:
        memory_bank = None
    losses = ["labels", "boxes", "rotate"]

    if args.rec:
        losses.append("rec")

    # english：38 or bilingual：4713
    if not args.is_bilingual:
        language = "LOWERCASE"
        charater = 38
    else:
        language = "CHINESE"
        charater = 4713

    criterion = ClipMatcher(
        num_classes,
        matcher=img_matcher,
        weight_dict=weight_dict,
        losses=losses,
        language=language,
    )
    criterion.to(device)
    postprocessors = {}
    model = TransDETR(
        backbone,
        transformer,
        track_embed=query_interaction_layer,
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        memory_bank=memory_bank,
        charater=charater,
    )
    return model, criterion, postprocessors
