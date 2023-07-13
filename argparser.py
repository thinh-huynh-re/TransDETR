import argparse
from typing import List, Optional
import numpy as np

from tap import Tap


def get_args_parser():
    parser = argparse.ArgumentParser("Deformable DETR Detector", add_help=False)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument(
        "--lr_backbone_names", default=["backbone.0"], type=str, nargs="+"
    )
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument(
        "--lr_linear_proj_names",
        default=[
            "reference_points",
            "sampling_offsets",
        ],
        type=str,
        nargs="+",
    )
    parser.add_argument("--is_bilingual", default=False, type=bool)
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr_drop", default=100, type=int)
    parser.add_argument("--save_period", default=50, type=int)
    parser.add_argument("--lr_drop_epochs", default=None, type=int, nargs="+")
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    parser.add_argument("--meta_arch", default="deformable_detr", type=str)

    parser.add_argument("--sgd", action="store_true")

    # Variants of Deformable DETR
    parser.add_argument("--with_box_refine", default=False, action="store_true")
    parser.add_argument("--two_stage", default=False, action="store_true")
    parser.add_argument("--accurate_ratio", default=False, action="store_true")

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )
    parser.add_argument("--num_anchors", default=1, type=int)

    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument("--enable_fpn", action="store_true")
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--position_embedding_scale",
        default=2 * np.pi,
        type=float,
        help="position / size * scale",
    )
    parser.add_argument(
        "--num_feature_levels", default=4, type=int, help="number of feature levels"
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=3,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=3,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=100, type=int, help="Number of query slots"
    )
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_n_points", default=4, type=int)
    parser.add_argument("--decoder_cross_self", default=False, action="store_true")
    parser.add_argument("--sigmoid_attn", default=False, action="store_true")
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--cj", action="store_true")
    parser.add_argument("--extra_track_attn", action="store_true")
    parser.add_argument("--loss_normalizer", action="store_true")

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # * recognition
    parser.add_argument(
        "--rec",
        action="store_true",
        help="Train recognition head if the flag is provided",
    )

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )

    # * Matcher
    parser.add_argument(
        "--mix_match",
        action="store_true",
    )
    parser.add_argument(
        "--set_cost_class",
        default=2,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--gt_file_train", type=str)
    parser.add_argument("--gt_file_val", type=str)
    parser.add_argument(
        "--coco_path", default="/data/workspace/detectron2/datasets/coco/", type=str
    )
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--pretrained", default=None, help="resume from checkpoint")
    parser.add_argument(
        "--cache_mode",
        default=False,
        action="store_true",
        help="whether to cache images on memory",
    )

    # end-to-end mot settings.
    parser.add_argument("--mot_path", default="/share/wuweijia/Data/MOT", type=str)
    parser.add_argument(
        "--data_txt_path_train",
        default="./datasets/data_path/detmot17.train",
        type=str,
        help="path to dataset txt split",
    )
    parser.add_argument(
        "--data_txt_path_val",
        default="./datasets/data_path/detmot17.train",
        type=str,
        help="path to dataset txt split",
    )

    parser.add_argument("--query_interaction_layer", default="QIM", type=str, help="")
    parser.add_argument("--sample_mode", type=str, default="fixed_interval")
    parser.add_argument("--sample_interval", type=int, default=1)
    parser.add_argument("--random_drop", type=float, default=0)
    parser.add_argument("--fp_ratio", type=float, default=0)
    parser.add_argument("--merger_dropout", type=float, default=0.1)
    parser.add_argument("--update_query_pos", action="store_true")

    parser.add_argument("--sampler_steps", type=int, nargs="*")
    parser.add_argument("--sampler_lengths", type=int, nargs="*")
    parser.add_argument("--exp_name", default="submit", type=str)
    parser.add_argument("--memory_bank_score_thresh", type=float, default=0.0)
    parser.add_argument("--memory_bank_len", type=int, default=4)
    parser.add_argument("--memory_bank_type", type=str, default="MemoryBank")
    parser.add_argument(
        "--memory_bank_with_self_attn", action="store_true", default=False
    )
    return parser


class ArgParser(Tap):
    lr: Optional[float] = 2e-4
    lr_backbone_names: Optional[List[str]] = ["backbone.0"]
    lr_backbone: Optional[float] = 2e-5
    lr_linear_proj_names: Optional[List[str]] = [
        "reference_points",
        "sampling_offsets",
    ]
    is_bilingual: Optional[bool] = False
    lr_linear_proj_mult: Optional[float] = 0.1
    batch_size: Optional[int] = 1
    weight_decay: Optional[float] = 1e-4
    epochs: Optional[int] = 50
    lr_drop: Optional[int] = 40
    save_period: Optional[int] = 50
    lr_drop_epochs: Optional[int] = None
    clip_max_norm: Optional[float] = 0.1
    meta_arch: Optional[str] = "TransDETR_ignored"
    sgd: Optional[bool] = False

    # Variants of Deformable DETR
    with_box_refine: Optional[bool] = True
    two_stage: Optional[bool] = False
    accurate_ratio: Optional[bool] = False

    # Model parameters
    frozen_weights: Optional[str] = None
    num_anchors: Optional[int] = 1

    # Backbone
    backbone: Optional[str] = "resnet50"
    enable_fpn: Optional[bool] = False
    dilation: Optional[bool] = False
    position_embedding: Optional[str] = "sine"  # choices=("sine", "learned")
    position_embedding_scale: Optional[float] = 2 * np.pi  # position / size * scale
    num_feature_levels: Optional[int] = 4

    # * Transformer
    enc_layers: Optional[int] = 3
    dec_layers: Optional[int] = 3
    dim_feedforward: Optional[int] = 1024
    hidden_dim: Optional[int] = 256
    dropout: Optional[float] = 0
    nheads: Optional[int] = 8
    num_queries: Optional[int] = 100  # !!!

    dec_n_points: Optional[int] = 4
    enc_n_points: Optional[int] = 4
    decoder_cross_self: Optional[bool] = False
    sigmoid_attn: Optional[bool] = False
    crop: Optional[bool] = False
    cj: Optional[bool] = False
    extra_track_attn: Optional[bool] = True
    loss_normalizer: Optional[bool] = False

    # * Segmentation
    masks: Optional[bool] = False

    # * recognition
    rec: Optional[bool] = False

    # Loss
    aux_loss: Optional[bool] = True

    # * Matcher
    mix_match: Optional[bool] = False
    set_cost_class: Optional[float] = 2
    set_cost_bbox: Optional[float] = 5
    set_cost_giou: Optional[float] = 2

    # Loss coefficients
    # skip

    # dataset parameters
    dataset_file: Optional[str] = "VideoText"
    gt_file_train: Optional[str] = None
    gt_file_val: Optional[str] = None
    coco_path: Optional[str] = "detectron2/datasets/coco/"
    coco_panoptic_path: Optional[str] = None
    remove_difficult: Optional[bool] = False

    output_dir: Optional[str] = "data/outputs/test"
    device: Optional[str] = "cuda"
    seed: Optional[int] = 42
    resume: Optional[str] = "weights/pretrain_COCOText_checkpoint.pth"
    start_epoch: Optional[int] = 0
    eval: Optional[bool] = False
    vis: Optional[bool] = False
    show: Optional[bool] = False
    num_workers: Optional[int] = 2
    pretrained: Optional[str] = None
    cache_mode: Optional[bool] = False

    mot_path: Optional[str] = "./data/frames/test"
    data_txt_path_train: Optional[str] = "./datasets/data_path/BOVText.train"
    data_txt_path_val: Optional[str] = "./datasets/data_path/BOVText.train"
    query_interaction_layer: Optional[str] = "QIM"
    sample_mode: Optional[str] = "random_interval"
    sample_interval: Optional[int] = 3
    random_drop: Optional[float] = 0.1
    fp_ratio: Optional[float] = 0.3
    merger_dropout: Optional[float] = 0.0
    update_query_pos: Optional[bool] = True

    sampler_steps: Optional[List[int]] = [50, 90, 120]
    sampler_lengths: Optional[List[int]] = [2, 3, 4, 5]
    exp_name: Optional[str] = "submit"
    memory_bank_score_thresh: Optional[float] = 0.0
    memory_bank_len: Optional[int] = 4
    memory_bank_type: Optional[str] = "MemoryBank"
    memory_bank_with_self_attn: Optional[bool] = False
