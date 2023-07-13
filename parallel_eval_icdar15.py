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
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
from PIL import Image, ImageDraw, ImageFont
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
import random
import argparse
import torchvision.transforms.functional as F
import torch
import json
import cv2
from pathlib import Path
from PIL import Image, ImageDraw
from models import build_model

from util.tool import load_model
from main import get_args_parser

from util.evaluation import Evaluator
from tqdm import tqdm
import math
import multiprocessing as mp

from detectron2.structures import Instances
from xml.dom.minidom import Document

try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
np.random.seed(2020)
from datasets.data_tools import get_vocabulary
from util.utils import write_lines
from collections import OrderedDict


def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    return img


class StorageDictionary(object):
    @staticmethod
    def dict2file(file_name, data_dict):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # import pickle
        output = open(file_name, "wb")
        pickle.dump(data_dict, output)
        output.close()

    @staticmethod
    def file2dict(file_name):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # import pickle
        pkl_file = open(file_name, "rb")
        data_dict = pickle.load(pkl_file)
        pkl_file.close()
        return data_dict

    @staticmethod
    def dict2file_json(file_name, data_dict):
        import json, io

        with io.open(file_name, "w", encoding="utf-8") as fp:
            # fp.write(unicode(json.dumps(data_dict, ensure_ascii=False, indent=4) ) )  #可以解决在文件里显示中文的问题，不加的话是 '\uxxxx\uxxxx'
            fp.write((json.dumps(data_dict, ensure_ascii=False, indent=4)))

    @staticmethod
    def file2dict_json(file_name):
        import json, io

        with io.open(file_name, "r", encoding="utf-8") as fp:
            data_dict = json.load(fp)
        return data_dict


def Generate_Json_annotation(TL_Cluster_Video_dict, Outpu_dir, xml_dir_):
    """ """
    ICDAR21_DetectionTracks = {}
    text_id = 1

    doc = Document()
    video_xml = doc.createElement("Frames")

    for frame in TL_Cluster_Video_dict.keys():
        doc.appendChild(video_xml)
        aperson = doc.createElement("frame")
        aperson.setAttribute("ID", str(frame))
        video_xml.appendChild(aperson)

        ICDAR21_DetectionTracks[frame] = []
        for text_list in TL_Cluster_Video_dict[frame]:
            #             ICDAR21_DetectionTracks[frame].append({"points":text_list[:8],"ID":text_list[8],"transcription":text_list[9],
            #                                                   "score":str(text_list[10]),
            #                                                   "roi_feature":text_list[11]})
            ICDAR21_DetectionTracks[frame].append(
                {
                    "points": text_list[:8],
                    "ID": text_list[8],
                    "transcription": text_list[9],
                }
            )
            # xml
            object1 = doc.createElement("object")
            object1.setAttribute("ID", str(text_list[8]))
            object1.setAttribute("Transcription", str(text_list[9]))
            #             object1.setAttribute("score", str(text_list[10]))
            aperson.appendChild(object1)

            for i in range(4):
                name = doc.createElement("Point")
                object1.appendChild(name)
                # personname = doc.createTextNode("1")
                name.setAttribute("x", str(int(text_list[i * 2])))
                name.setAttribute("y", str(int(text_list[i * 2 + 1])))

    StorageDictionary.dict2file_json(Outpu_dir, ICDAR21_DetectionTracks)

    # xml
    f = open(xml_dir_, "w")
    f.write(doc.toprettyxml(indent="  "))
    f.close()


def is_chinese(string):
    """
    check is chinese
    :param string:
    :return: bool
    """
    for ch in string:
        if "\u4e00" <= ch <= "\u9fff":
            return True

    return False


def cv2AddChineseText(image, text, position, textColor=(0, 0, 0), textSize=30):
    x1, y1 = position
    x2, y2 = len(text) * textSize / 2 + x1, y1 + textSize
    if is_chinese(text):
        x2, y2 = len(text) * textSize + x1, y1 + textSize

    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
    mask_1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask_1, [points], 1)

    image, rgb = mask_image_bg(image, mask_1, rgb=[0, 0, 0])

    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(image)

    fontStyle = ImageFont.truetype("./tools/simsun.ttc", textSize, encoding="utf-8")

    draw.text(position, text, textColor, font=fontStyle)

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    return image


def mask_image_bg(image, mask_2d, rgb=None, valid=False):
    h, w = mask_2d.shape

    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")

    image.astype("uint8")
    mask = (mask_2d != 0).astype(bool)
    if rgb is None:
        rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)

    mask_3d_color[mask_2d[:, :] == 1] = rgb
    image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5

    if valid:
        mask_3d_color[mask_2d[:, :] == 1] = [[0, 0, 0]]
        kernel = np.ones((5, 5), np.uint8)
        mask_2d = cv2.dilate(mask_2d, kernel, iterations=4)
        mask = (mask_2d != 0).astype(bool)
        image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5
        return image, rgb

    return image, rgb


def mask_image(image, mask_2d, rgb=None, valid=False):
    h, w = mask_2d.shape

    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")

    image.astype("uint8")
    mask = (mask_2d != 0).astype(bool)
    if rgb is None:
        rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)

    mask_3d_color[mask_2d[:, :] == 1] = rgb
    image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5

    if valid:
        mask_3d_color[mask_2d[:, :] == 1] = [[0, 0, 0]]
        kernel = np.ones((5, 5), np.uint8)
        mask_2d = cv2.dilate(mask_2d, kernel, iterations=4)
        mask = (mask_2d != 0).astype(bool)
        image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5
        return image, rgb

    return image, rgb


def draw_bboxes_gt(
    ori_img,
    annotatation_frame,
    identities=None,
    offset=(0, 0),
    cvt_color=False,
    rgbs=None,
):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img

    for data in annotatation_frame:
        x1, y1, x2, y2, x3, y3, x4, y4 = data["points"]
        ID = data["ID"]
        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        id_content = str(data["transcription"])

        if id_content == "###":
            cv2.polylines(img, [points], True, (0, 0, 255), thickness=5)
        else:
            cv2.polylines(img, [points], True, (255, 0, 0), thickness=5)

    return img


def draw_bboxes(
    ori_img,
    bbox,
    words,
    scores,
    identities=None,
    offset=(0, 0),
    cvt_color=False,
    rgbs=None,
):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2, x3, y3, x4, y4 = [int(i) for i in box[:8]]

        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        mask_1 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask_1, [points], 1)

        ID = int(identities[i]) if identities is not None else 0
        word = words[i]
        score = str(np.array(scores[i]))[:4]
        if ID in rgbs:
            img, rgb = mask_image(img, mask_1, rgbs[ID])
        else:
            img, rgb = mask_image(img, mask_1)
            rgbs[ID] = rgb
        r, g, b = rgb[0]
        r, g, b = int(r), int(g), int(b)
        cv2.polylines(img, [points], True, (r, g, b), thickness=4)
        #         img=cv2AddChineseText(img,str(ID), (int(x1), int(y1) - 20),((0,0,255)), 45)
        short_side = min(img.shape[0], img.shape[1])
        text_size = int(short_side * 0.03)

        img = cv2AddChineseText(
            img,
            str(word) + "|" + score,
            (int(x1), int(y1) - text_size),
            ((255, 255, 255)),
            text_size,
        )
    return img


def draw_points(
    img: np.ndarray, points: np.ndarray, color=(255, 255, 255)
) -> np.ndarray:
    assert (
        len(points.shape) == 2 and points.shape[1] == 2
    ), "invalid points shape: {}".format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 2, color=color, thickness=2)
    return img


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class Track(object):
    track_cnt = 0

    def __init__(self, box):
        self.box = box
        self.time_since_update = 0
        self.id = Track.track_cnt
        Track.track_cnt += 1
        self.miss = 0

    def miss_one_frame(self):
        self.miss += 1

    def clear_miss(self):
        self.miss = 0

    def update(self, box):
        self.box = box
        self.clear_miss()


class MOTR(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        pass

    def update(self, dt_instances: Instances):
        ret = []
        for i in range(len(dt_instances)):
            label = dt_instances.labels[i]
            if label == 0:
                id = dt_instances.obj_idxes[i]
                box_with_score = np.concatenate(
                    [dt_instances.boxes[i], dt_instances.scores[i : i + 1]], axis=-1
                )
                ret.append(
                    np.concatenate((box_with_score, [id + 1])).reshape(1, -1)
                )  # +1 as MOT benchmark requires positive

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))


def load_label(label_path: str, img_size: tuple) -> dict:
    labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
    h, w = img_size
    # Normalized cewh to pixel xyxy format
    labels = labels0.copy()
    labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4] / 2)
    labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5] / 2)
    labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4] / 2)
    labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5] / 2)
    targets = {"boxes": [], "labels": [], "area": []}
    num_boxes = len(labels)

    visited_ids = set()
    for label in labels[:num_boxes]:
        obj_id = label[1]
        if obj_id in visited_ids:
            continue
        visited_ids.add(obj_id)
        targets["boxes"].append(label[2:6].tolist())
        targets["area"].append(label[4] * label[5])
        targets["labels"].append(0)
    targets["boxes"] = np.asarray(targets["boxes"])
    targets["area"] = np.asarray(targets["area"])
    targets["labels"] = np.asarray(targets["labels"])
    return targets


def filter_pub_det(res_file, pub_det_file, filter_iou=False):
    frame_boxes = {}
    with open(pub_det_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if len(line) == 0:
                continue
            elements = line.strip().split(",")
            frame_id = int(elements[0])
            x1, y1, w, h = elements[2:6]
            x1, y1, w, h = float(x1), float(y1), float(w), float(h)
            x2 = x1 + w - 1
            y2 = y1 + h - 1
            if frame_id not in frame_boxes:
                frame_boxes[frame_id] = []
            frame_boxes[frame_id].append([x1, y1, x2, y2])

    for frame, boxes in frame_boxes.items():
        frame_boxes[frame] = np.array(boxes)

    ids = {}
    num_filter_box = 0
    with open(res_file, "r") as f:
        lines = list(f.readlines())
    with open(res_file, "w") as f:
        for line in lines:
            if len(line) == 0:
                continue

            elements = line.strip().split(",")
            frame_id, obj_id = elements[:2]
            frame_id = int(frame_id)
            obj_id = int(obj_id)
            x1, y1, w, h = elements[2:6]
            x1, y1, w, h = float(x1), float(y1), float(w), float(h)
            x2 = x1 + w - 1
            y2 = y1 + h - 1
            if obj_id not in ids:
                # track initialization.
                if frame_id not in frame_boxes:
                    num_filter_box += 1
                    print("filter init box {} {}".format(frame_id, obj_id))
                    continue
                pub_dt_boxes = frame_boxes[frame_id]
                dt_box = np.array([[x1, y1, x2, y2]])
                if filter_iou:
                    max_iou = bbox_iou(dt_box, pub_dt_boxes).max()
                    if max_iou < 0.5:
                        num_filter_box += 1
                        print("filter init box {} {}".format(frame_id, obj_id))
                        continue
                else:
                    pub_dt_centers = (pub_dt_boxes[:, :2] + pub_dt_boxes[:, 2:4]) * 0.5
                    x_inside = (dt_box[0, 0] <= pub_dt_centers[:, 0]) & (
                        dt_box[0, 2] >= pub_dt_centers[:, 0]
                    )
                    y_inside = (dt_box[0, 1] <= pub_dt_centers[:, 1]) & (
                        dt_box[0, 3] >= pub_dt_centers[:, 1]
                    )
                    center_inside: np.ndarray = x_inside & y_inside
                    if not center_inside.any():
                        num_filter_box += 1
                        print("filter init box {} {}".format(frame_id, obj_id))
                        continue
                print("save init track {} {}".format(frame_id, obj_id))
                ids[obj_id] = True
            f.write(line)

    print("totally {} boxes are filtered.".format(num_filter_box))


def get_rotate_mat(theta):
    """positive theta value means rotate clockwise"""
    return np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )


def load_img_from_file(f_path):
    label_path = (
        f_path.replace("images", "labels_with_ids")
        .replace(".png", ".txt")
        .replace(".jpg", ".txt")
    )
    cur_img = cv2.imread(f_path)
    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)

    targets = (
        load_label(label_path, cur_img.shape[:2])
        if os.path.exists(label_path)
        else None
    )
    return cur_img, targets


class Detector(object):
    def __init__(self, args, model=None, seq_num=2):
        self.args = args
        self.detr = model

        self.seq_num = seq_num
        img_list = os.listdir(os.path.join(self.args.mot_path, self.seq_num))
        img_list = [_ for _ in img_list if ("jpg" in _) or ("png" in _)]

        if "YVT" in args.data_txt_path_val:
            self.img_list = [
                os.path.join(
                    self.args.mot_path,
                    self.seq_num,
                    "{}f{}.jpg".format(self.seq_num, str(_).zfill(4)),
                )
                for _ in range(0, len(img_list))
            ]
            # rec  CHINESE  LOWERCASE
            voc, char2id, id2char = get_vocabulary("LOWERCASE", use_ctc=True)
        elif "minetto" in args.data_txt_path_val:
            self.img_list = [
                os.path.join(
                    self.args.mot_path, self.seq_num, "{}.jpg".format(str(_).zfill(6))
                )
                for _ in range(0, len(img_list))
            ]
            voc, char2id, id2char = get_vocabulary("LOWERCASE", use_ctc=True)
        elif "BOVText" in args.data_txt_path_val:
            self.img_list = [
                os.path.join(self.args.mot_path, self.seq_num, "{}.jpg".format(_))
                for _ in range(1, len(img_list) + 1)
            ]
            voc, char2id, id2char = get_vocabulary("CHINESE", use_ctc=True)
        elif "TextVR" in args.data_txt_path_val:
            self.img_list = [
                os.path.join(
                    self.args.mot_path, self.seq_num, "{}".format(_).zfill(8) + ".jpg"
                )
                for _ in range(0, len(img_list))
            ]
            voc, char2id, id2char = get_vocabulary("LOWERCASE", use_ctc=True)
        elif "DSText" in args.data_txt_path_val:
            self.img_list = [
                os.path.join(self.args.mot_path, self.seq_num, "{}.jpg".format(_))
                for _ in range(1, len(img_list) + 1)
            ]
            voc, char2id, id2char = get_vocabulary("LOWERCASE", use_ctc=True)
        else:
            self.img_list = [
                os.path.join(self.args.mot_path, self.seq_num, "{}.jpg".format(_))
                for _ in range(1, len(img_list) + 1)
            ]
            voc, char2id, id2char = get_vocabulary("LOWERCASE", use_ctc=True)

        try:
            self.ann = self.get_annotation(
                "./tools/Evaluation_ICDAR13/gt/{}_GT.json".format(self.seq_num)
            )
        except:
            self.ann = None

        self.img_len = len(self.img_list)
        self.tr_tracker = MOTR()

        # 解码使用
        self.char2id = char2id
        self.id2char = id2char
        self.blank = char2id["PAD"]

        """
        common settings
        """
        self.img_height = 800
        self.img_width = 1536

        # BOVText
        self.img_height = 640
        self.img_width = 1536

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.save_path = os.path.join(
            self.args.output_dir, "results/{}".format(seq_num)
        )
        os.makedirs(self.save_path, exist_ok=True)

        predict_path = os.path.join(self.args.output_dir, "preds_3")
        os.makedirs(predict_path, exist_ok=True)

        if "minetto" in args.data_txt_path_val:
            xml_name = self.seq_num
        elif "BOVText" in args.data_txt_path_val:
            self.seq_num = self.seq_num.replace("/", "_")
            xml_name = self.seq_num

            self.predict_path = os.path.join(
                predict_path, "res_{}.xml".format(xml_name)
            )

            json_path = os.path.join(self.args.output_dir, "jons_3")
            os.makedirs(json_path, exist_ok=True)
            self.json_path = os.path.join(
                json_path, "{}.json".format(self.seq_num.split("/")[-1])
            )

        elif "DSText" in args.data_txt_path_val:
            xml_name = self.seq_num.split("/")[-1]
            self.predict_path = os.path.join(
                predict_path, "res_{}.xml".format(xml_name)
            )

            json_path = os.path.join(self.args.output_dir, "jons")
            os.makedirs(json_path, exist_ok=True)
            self.json_path = os.path.join(
                json_path, "{}.json".format(self.seq_num.split("/")[-1])
            )
        else:
            xml_name = self.seq_num.split("_")
            xml_name = xml_name[0] + "_" + xml_name[1]
            self.predict_path = os.path.join(
                predict_path, "res_{}.xml".format(xml_name.replace("V", "v"))
            )

            json_path = os.path.join(self.args.output_dir, "jons")
            os.makedirs(json_path, exist_ok=True)
            self.json_path = os.path.join(json_path, "{}.json".format(self.seq_num))

    def get_annotation(self, video_path):
        annotation = {}
        with open(video_path, "r", encoding="utf-8-sig") as load_f:
            gt = json.load(load_f)
        for child in gt:
            lines = gt[child]
            annotation.update({child: lines})
        return annotation

    def init_img(self, img):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.obj_idxes >= 0
        dt_instances = dt_instances[keep]
        #         keep = dt_instances.scores > prob_threshold

        return dt_instances

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        dt_instances = dt_instances[keep]

        return dt_instances

    @staticmethod
    def write_results(txt_path, frame_id, bbox_xyxy, identities):
        save_format = "{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n"
        with open(txt_path, "a") as f:
            for xyxy, track_id in zip(bbox_xyxy, identities):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                line = save_format.format(
                    frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h
                )
                f.write(line)

    def eval_seq(self):
        data_root = os.path.join(
            self.args.mot_path, "/share/wuweijia/Data/MOT/MOT15/images/train"
        )
        result_filename = os.path.join(self.predict_path, "gt.txt")
        evaluator = Evaluator(data_root, self.seq_num)
        accs = evaluator.eval_file(result_filename)
        return accs

    @staticmethod
    def visualize_img_with_bbox(
        img_path, img, dt_instances: Instances, ref_pts=None, gt_boxes=None, rgbs=None
    ):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if dt_instances.has("scores"):
            img_show = draw_bboxes(
                img,
                dt_instances.boxes,
                dt_instances.word,
                dt_instances.scores,
                dt_instances.obj_idxes,
                rgbs=rgbs,
            )
        #         else:
        #             img_show = draw_bboxes(img, dt_instances.boxes,dt_instances.scores, dt_instances.obj_idxes,rgbs=rgbs)

        if ref_pts is not None:
            img_show = draw_points(img_show, ref_pts)

        if gt_boxes is not None:
            img_show = draw_bboxes_gt(img_show, gt_boxes)

        cv2.imwrite(img_path, img_show)

    #     @staticmethod
    def to_rotated_rec(
        self, dt_instances: Instances, filter_word_score=0.5
    ) -> Instances:
        out_rec_decoded = dt_instances.word
        preds_max_prob = dt_instances.word_max_prob
        words = []
        num_words = out_rec_decoded.size(0)
        word_scores = []
        for l in range(num_words):
            s = ""
            num_chars = 0
            c_word_score = 0.0
            word_preds_max_prob = preds_max_prob[l]
            t = out_rec_decoded[l]  # 32
            for i in range(len(t)):
                if t[i].item() != self.blank:
                    c_word_score += word_preds_max_prob[i]
                    num_chars += 1
                    if not (
                        i > 0 and t[i - 1].item() == t[i].item()
                    ):  # removing repeated characters and blank.
                        s += self.id2char[t[i].item()]

            word_scores.append(c_word_score / (num_chars + 0.000001))
            words.append(s)

        dt_instances.word = words
        #         word_scores = torch.as_tensor(np.array(word_scores))
        #         dt_instances.word_max_prob = word_scores
        #         keep = dt_instances.scores>filter_word_score
        #         dt_instances = dt_instances[keep]

        boxes = []
        for box, angle in zip(dt_instances.boxes, dt_instances.rotate):
            x_min, y_min, x_max, y_max = [int(i) for i in box[:4]]
            rotate = angle
            rotate_mat = get_rotate_mat(-rotate)
            temp_x = np.array([[x_min, x_max, x_max, x_min]]) - (x_min + x_max) / 2
            temp_y = np.array([[y_min, y_min, y_max, y_max]]) - (y_min + y_max) / 2
            coordidates = np.concatenate((temp_x, temp_y), axis=0)
            res = np.dot(rotate_mat, coordidates)
            res[0, :] += (x_min + x_max) / 2
            res[1, :] += (y_min + y_max) / 2
            boxes.append(
                np.array(
                    [
                        res[0, 0],
                        res[1, 0],
                        res[0, 1],
                        res[1, 1],
                        res[0, 2],
                        res[1, 2],
                        res[0, 3],
                        res[1, 3],
                    ]
                )
            )
        dt_instances.boxes = np.array(boxes)

        return dt_instances

    def detect(self, time_cost={}, prob_threshold=0.1, area_threshold=5, vis=False):
        total_dts = 0
        track_instances = None
        max_id = 0
        rgbs = {}
        annotation = {}

        dict_one_cost = {
            "backbone_time": 0,
            "nect_time": 0,
            "upsample_time": 0,
            "transformer_time": 0,
            "det_head_time": 0,
            "rec_head_time": 0,
            "memory_embed_time": 0,
            "postprocess_time": 0,
        }

        #         if os.path.exists(self.json_path):
        #             return time_cost

        for i in tqdm(range(0, self.img_len)):
            img, targets = load_img_from_file(self.img_list[i])

            cur_img, ori_img = self.init_img(img)

            # track_instances = None
            if track_instances is not None:
                track_instances.remove("boxes")
                track_instances.remove("labels")
                track_instances.remove("rotate")
                track_instances.remove("word")
                track_instances.remove("word_max_prob")
                track_instances.remove("roi")

            res, time_cost_frame = self.detr.inference_single_image(
                cur_img.cuda().float(), (self.seq_h, self.seq_w), track_instances
            )

            #             time_cost["backbone_time"]+= time_cost_frame["backbone_time"]
            #             time_cost["nect_time"]+= time_cost_frame["nect_time"]
            #             time_cost["upsample_time"]+= time_cost_frame["upsample_time"]
            #             time_cost["det_head_time"]+= time_cost_frame["det_head_time"]
            #             time_cost["transformer_time"]+= time_cost_frame["transformer_time"]
            #             time_cost["rec_head_time"]+= time_cost_frame["rec_head_time"]
            #             time_cost["memory_embed_time"]+= time_cost_frame["memory_embed_time"]
            #             time_cost["postprocess_time"]+= time_cost_frame["postprocess_time"]

            #             dict_one_cost["backbone_time"]+= time_cost_frame["backbone_time"]
            #             dict_one_cost["nect_time"]+= time_cost_frame["nect_time"]
            #             dict_one_cost["upsample_time"]+= time_cost_frame["upsample_time"]
            #             dict_one_cost["det_head_time"]+= time_cost_frame["det_head_time"]
            #             dict_one_cost["transformer_time"]+= time_cost_frame["transformer_time"]
            #             dict_one_cost["rec_head_time"]+= time_cost_frame["rec_head_time"]
            #             dict_one_cost["memory_embed_time"]+= time_cost_frame["memory_embed_time"]
            #             dict_one_cost["postprocess_time"]+= time_cost_frame["postprocess_time"]

            track_instances = res["track_instances"]
            max_id = max(max_id, track_instances.obj_idxes.max().item())

            all_ref_pts = tensor_to_numpy(res["ref_pts"][0, :, :2])
            dt_instances = track_instances.to(torch.device("cpu"))

            short_side = min(self.seq_h, self.seq_w)
            area_threshold = int(short_side * 0.02) * int(short_side * 0.02)

            # filter det instances by score.
            #             dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)
            dt_instances = self.to_rotated_rec(dt_instances)

            total_dts += len(dt_instances)

            if vis:
                # for visual
                cur_vis_img_path = os.path.join(self.save_path, "{}.jpg".format(i))
                if self.ann == None:
                    gt_boxes = None
                else:
                    gt_boxes = self.ann[str(i + 1)]
                self.visualize_img_with_bbox(
                    cur_vis_img_path,
                    ori_img,
                    dt_instances,
                    ref_pts=all_ref_pts,
                    gt_boxes=gt_boxes,
                    rgbs=rgbs,
                )

            boxes, IDs, scores, words = (
                dt_instances.boxes,
                dt_instances.obj_idxes,
                dt_instances.scores,
                dt_instances.word,
            )
            roi_features = dt_instances.roi
            lines = []
            for box, ID, score, word, roi_feature in zip(
                boxes, IDs, scores, words, roi_features
            ):
                score = score.item()
                roi_feature = np.array(roi_feature).tolist()
                x1, y1, x2, y2, x3, y3, x4, y4 = [int(i) for i in box[:8]]

                if (
                    score < 0.83
                    and self.seq_num == "Cls6_NewsReport_Cls6_NewsReport_video76"
                ):
                    continue

                lines.append(
                    [x1, y1, x2, y2, x3, y3, x4, y4, int(ID), word, score, roi_feature]
                )

            annotation.update({str(i + 1): lines})
        Generate_Json_annotation(annotation, self.json_path, self.predict_path)

        print("totally {} dts max_id={}".format(total_dts, max_id))
        return time_cost


def getBboxesAndLabels_icd131(annotations):
    bboxes = []
    labels = []
    polys = []
    bboxes_ignore = []
    labels_ignore = []
    polys_ignore = []
    Transcriptions = []
    IDs = []
    rotates = []
    confidences = []
    # points_lists = [] # does not contain the ignored polygons.
    for annotation in annotations:
        object_boxes = []
        for point in annotation:
            object_boxes.append([int(point.attrib["x"]), int(point.attrib["y"])])

        points = np.array(object_boxes).reshape((-1))
        points = cv2.minAreaRect(points.reshape((4, 2)))
        points = cv2.boxPoints(points).reshape((-1))
        IDs.append(annotation.attrib["ID"])
        Transcriptions.append(annotation.attrib["Transcription"])
        #         confidences.append(annotation.attrib["score"])
        confidences.append(1)
        bboxes.append(points)

    if bboxes:
        IDs = np.array(IDs, dtype=np.int64)
        bboxes = np.array(bboxes, dtype=np.float32)
    else:
        bboxes = np.zeros((0, 8), dtype=np.float32)
        IDs = np.array([], dtype=np.int64)
        Transcriptions = []
        confidences = []

    return bboxes, IDs, Transcriptions, confidences


def parse_xml_rec(annotation_path):
    utf8_parser = ET.XMLParser(encoding="gbk")
    with open(annotation_path, "r", encoding="gbk") as load_f:
        tree = ET.parse(load_f, parser=utf8_parser)
    root = tree.getroot()

    ann_dict = {}
    for idx, child in enumerate(root):
        bboxes, IDs, Transcriptions, confidences = getBboxesAndLabels_icd131(child)
        ann_dict[child.attrib["ID"]] = [bboxes, IDs, Transcriptions, confidences]
    return ann_dict


# OrderedDict
def sort_key(old_dict, reverse=False):
    keys = [int(i) for i in old_dict.keys()]
    keys = sorted(keys, reverse=reverse)

    new_dict = OrderedDict()

    for key in keys:
        new_dict[str(key)] = old_dict[str(key)]
    return new_dict


def getid_text(new_xml_dir_):
    voc_dict = {
        "res_video_11.xml": "Video_11_4_1_GT_voc.txt",
        "res_video_15.xml": "Video_15_4_1_GT_voc.txt",
        "res_video_17.xml": "Video_17_3_1_GT_voc.txt",
        "res_video_1.xml": "Video_1_1_2_GT_voc.txt",
        "res_video_20.xml": "Video_20_5_1_GT_voc.txt",
        "res_video_22.xml": "Video_22_5_1_GT_voc.txt",
        "res_video_23.xml": "Video_23_5_2_GT_voc.txt",
        "res_video_24.xml": "Video_24_5_2_GT_voc.txt",
        "res_video_30.xml": "Video_30_2_3_GT_voc.txt",
        "res_video_32.xml": "Video_32_2_3_GT_voc.txt",
        "res_video_34.xml": "Video_34_2_3_GT_voc.txt",
        "res_video_35.xml": "Video_35_2_3_GT_voc.txt",
        "res_video_38.xml": "Video_38_2_3_GT_voc.txt",
        "res_video_39.xml": "Video_39_2_3_GT_voc.txt",
        "res_video_43.xml": "Video_43_6_4_GT_voc.txt",
        "res_video_44.xml": "Video_44_6_4_GT_voc.txt",
        "res_video_48.xml": "Video_48_6_4_GT_voc.txt",
        "res_video_49.xml": "Video_49_6_4_GT_voc.txt",
        "res_video_50.xml": "Video_50_7_4_GT_voc.txt",
        "res_video_53.xml": "Video_53_7_4_GT_voc.txt",
        "res_video_55.xml": "Video_55_3_2_GT_voc.txt",
        "res_video_5.xml": "Video_5_3_2_GT_voc.txt",
        "res_video_6.xml": "Video_6_3_2_GT_voc.txt",
        "res_video_9.xml": "Video_9_1_1_GT_voc.txt",
    }

    for xml in tqdm(os.listdir(new_xml_dir_)):
        id_trans = {}
        id_cond = {}
        if ".txt" in xml or "ipynb" in xml:
            continue

        lines = []
        xml_one = os.path.join(new_xml_dir_, xml)
        ann = parse_xml_rec(xml_one)
        for frame_id_ann in ann:
            points, IDs, Transcriptions, confidences = ann[frame_id_ann]
            for ids, trans, confidence in zip(IDs, Transcriptions, confidences):
                if str(ids) in id_trans:
                    id_trans[str(ids)].append(trans)
                    id_cond[str(ids)].append(float(confidence))
                else:
                    id_trans[str(ids)] = [trans]
                    id_cond[str(ids)] = [float(confidence)]

        id_trans = sort_key(id_trans)
        id_cond = sort_key(id_cond)

        for i in id_trans:
            txts = id_trans[i]
            confidences = id_cond[i]
            txt = max(txts, key=txts.count)

            #             sco = 0
            #             txt = txts[0]
            #             for txt1,confident in zip(txts,confidences):
            #                 if confident>sco:
            #                     sco = confident
            #                     txt = txt1

            lines.append('"' + i + '"' + "," + '"' + txt + '"' + "\n")
        write_lines(os.path.join(new_xml_dir_, xml.replace("xml", "txt")), lines)


def sub_processor(pid, args, video_list):
    torch.cuda.set_device(pid)
    # load model and weights
    detr, _, _ = build_model(args)
    checkpoint = torch.load(args.resume, map_location="cpu")
    detr = load_model(detr, args.resume)
    detr = detr.cuda()
    detr.eval()

    text = "processor %d" % pid

    # 1. For each video
    for video in video_list:
        print(video)
        det = Detector(args, model=detr, seq_num=video)
        time_cost = det.detect()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if "ICDAR15" in args.data_txt_path_val:
        args.mot_path = os.path.join(args.mot_path, "ICDAR2015/images/test")
        seq_nums = os.listdir(args.mot_path)

    elif "DSText" in args.data_txt_path_val:
        args.mot_path = os.path.join(args.mot_path, "DSText/images/test")
        seq_nums = []
        for seq in os.listdir(args.mot_path):
            for video_name in os.listdir(os.path.join(args.mot_path, seq)):
                seq_nums.append(os.path.join(seq, video_name))

    elif "TextVR" in args.data_txt_path_val:
        args.mot_path = (
            "/share/mmu-ocr/datasets/zyz_anns/frozen-in-time-ocr/cache/vitvr"
        )
        seq_nums = os.listdir(args.mot_path)

    elif "YVT" in args.data_txt_path_val:
        args.mot_path = os.path.join(args.mot_path, "YVT/images/test")
        seq_nums = os.listdir(args.mot_path)

    elif "minetto" in args.data_txt_path_val:
        args.mot_path = os.path.join(args.mot_path, "minetto/images/test")
        seq_nums = os.listdir(args.mot_path)

    elif "BOVText" in args.data_txt_path_val:
        args.mot_path = "/share/wuweijia/MyBenchMark/MMVText/BOVTextV2/Test/Frames"
        seq_nums = []
        for seq in os.listdir(args.mot_path):
            for video_name in os.listdir(os.path.join(args.mot_path, seq)):
                selected_file = [
                    #                     "Cls27_Education_Cls27_Education_video56"
                    "Cls16_Government_Cls16_Government_video15",
                    "Cls1_Livestreaming_Cls1_Livestreaming_video39",
                    "Cls24_Fishery_Cls24_Fishery_video79",
                    "Cls24_Fishery_Cls24_Fishery_video75",
                    "Cls13_Introduction_Cls13_Introduction_video30",
                    "Cls7_Game_Cls7_Game_video10",
                    "Cls14_Talent_Cls14_Talent_video27",
                    "Cls14_Talent_Cls14_Talent_video44",
                    "Cls14_Talent_Cls14_Talent_video55",
                    "Cls14_Talent_Cls14_Talent_video46",
                    "Cls26_Technology_Cls26_Technology_video1",
                    "Cls20_Campus_Cls20_Campus_video7",
                    "Cls6_NewsReport_Cls6_NewsReport_video76",
                    "Cls16_Government_Cls16_Government_video15",
                    "Cls3_Sports_Cls3_Sports_video47.json",
                    "Cls28_BeautyIndustry_Cls28_BeautyIndustry_video36",
                    "Cls6_NewsReport_Cls6_NewsReport_video26",
                    "Cls23_International_Cls23_International_video26",
                    "Cls17_Speech_Cls17_Speech_video12",
                    "Cls17_Speech_Cls17_Speech_video53",
                ]

                # fp高
                #                 selected_file = [
                # #                  "Cls6_NewsReport_Cls6_NewsReport_video76"
                # #                   ,
                #                     "Cls26_Technology_Cls26_Technology_video1"
                #                 ]
                print(seq + "_" + video_name)

                if seq + "_" + video_name not in selected_file:
                    continue

                seq_nums.append(os.path.join(seq, video_name))
    else:
        raise NotImplementedError()

    accs = []
    seqs = []

    ICDAR2013_seqs = [
        "Video_20_5_1",
        "Video_6_3_2",
        "Video_49_6_4",
        "Video_5_3_2",
        "Video_32_2_3",
        "Video_23_5_2",
        "Video_39_2_3",
        "Video_35_2_3",
        "Video_1_1_2",
        "Video_44_6_4",
        "Video_17_3_1",
        "Video_24_5_2",
        "Video_11_4_1",
        "Video_53_7_4",
        "Video_48_6_4",
    ]
    test_seqs = ["Video_11_4_1"]
    dict_cost = {
        "backbone_time": 0,
        "nect_time": 0,
        "upsample_time": 0,
        "transformer_time": 0,
        "det_head_time": 0,
        "rec_head_time": 0,
        "memory_embed_time": 0,
        "postprocess_time": 0,
    }

    result_dict = mp.Manager().dict()
    mp = mp.get_context("spawn")
    thread_num = args.thread_num
    processes = []
    per_thread_video_num = int(len(seq_nums) / thread_num)

    print("Start inference")
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = seq_nums[i * per_thread_video_num :]
        else:
            sub_video_list = seq_nums[
                i * per_thread_video_num : (i + 1) * per_thread_video_num
            ]

        p = mp.Process(target=sub_processor, args=(i, args, sub_video_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    result_dict = dict(result_dict)

    if "ICDAR15" in args.data_txt_path_val:
        getid_text(os.path.join(args.output_dir, "preds_3"))
        print(dict_cost)
