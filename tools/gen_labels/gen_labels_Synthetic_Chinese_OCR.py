import os.path as osp
import os
import numpy as np
import json
from util.utils import write_result_as_txt, debug, setup_logger, write_lines, MyEncoder

try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import math
from tqdm import tqdm


def adjust_box_sort(box):
    start = -1
    _box = list(np.array(box).reshape(-1, 2))
    min_x = min(box[0::2])
    min_y = min(box[1::2])
    _box.sort(key=lambda x: (x[0] - min_x) ** 2 + (x[1] - min_y) ** 2)
    start_point = list(_box[0])
    for i in range(0, 8, 2):
        x, y = box[i], box[i + 1]
        if [x, y] == start_point:
            start = i // 2
            break

    new_box = []
    new_box.extend(box[start * 2 :])
    new_box.extend(box[: start * 2])
    return np.array(new_box)


def find_min_rect_angle(vertices):
    """find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    """

    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    vertices = adjust_box_sort(vertices)
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * (
            max(y1, y2, y3, y4) - min(y1, y2, y3, y4)
        )
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float("inf")
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def rotate_vertices(vertices, theta, anchor=None):
    """rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    """
    v = vertices.reshape((4, 2)).T
    #     print(v)
    #     print(anchor)
    if anchor is None:
        #         anchor = v[:, :1]
        anchor = np.array([[v[0].sum()], [v[1].sum()]]) / 4
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def get_rotate_mat(theta):
    """positive theta value means rotate clockwise"""
    return np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )


def cal_error(vertices):
    """default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    """
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = (
        cal_distance(x1, y1, x_min, y_min)
        + cal_distance(x2, y2, x_max, y_min)
        + cal_distance(x3, y3, x_max, y_max)
        + cal_distance(x4, y4, x_min, y_max)
    )
    return err


def get_boundary(vertices):
    """get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def cal_distance(x1, y1, x2, y2):
    """calculate the Euclidean distance"""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_rotate(box):
    # box : x1,y2...,x3,y3
    theta = find_min_rect_angle(box)

    rotate_mat = get_rotate_mat(theta)
    rotated_vertices = rotate_vertices(box, theta)
    x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
    return np.array([x_min, y_min, x_max - x_min, y_max - y_min]), theta


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def gen_data_path(
    path,
    split_train_test="train",
    data_path_str="./datasets/data_path/Synthetic_Chinese_OCR.train",
):
    image_path = os.path.join(path, "images", split_train_test)
    lines = []
    for video_name in os.listdir(image_path):
        frame_path = os.path.join(image_path, video_name)
        frame_list = []
        for frame_path_ in os.listdir(frame_path):
            if ".jpg" in frame_path_:
                frame_list.append(frame_path_)

        for i in frame_list:
            label_path = "/mmu-ocr/weijiawu/Data/VideoText/MOTR/Synthetic_Chinese_OCR/labels_with_ids/train/images/" + i.replace(
                "jpg", "txt"
            ).replace(
                "png", "txt"
            )
            if not os.path.exists(label_path):
                continue
            #                 with open(label_path, 'w') as f:
            #                     pass

            frame_real_path = (
                "Synthetic_Chinese_OCR/images/train/"
                + video_name
                + "/{}".format(i)
                + "\n"
            )
            lines.append(frame_real_path)
    write_lines(data_path_str, lines)


from_label_root = "/mmu-ocr/weijiawu/Data/VideoText/MOTR/Synthetic_Chinese_OCR/images"
seq_root = "/mmu-ocr/weijiawu/Data/VideoText/MOTR/Synthetic_Chinese_OCR/images/train"
label_root = (
    "/mmu-ocr/weijiawu/Data/VideoText/MOTR/Synthetic_Chinese_OCR/labels_with_ids/train"
)
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]


tid_curr = 0
tid_last = -1
for seq in seqs:
    image_path_frame = osp.join(seq_root, seq)
    seq_label_root = osp.join(label_root, seq)
    print(seq_label_root)
    mkdirs(seq_label_root)

    ann_path = os.path.join(from_label_root, "data_train.txt")
    char_std_path = os.path.join(from_label_root, "char_std_5990.txt")

    char_dict = {}
    with open(char_std_path, "r") as file:
        for i, line in enumerate(file):
            char_dict[str(i)] = line.strip()

    lines = []
    with open(ann_path, "r") as file:
        for line in file:
            lines.append(line.strip())

    for idx, a in tqdm(enumerate(lines)):
        filename = a.split(" ")[0]
        text = a.split(" ")[1:]

        text_char = ""
        for cc in text:
            text_char = text_char + char_dict[cc]

        one_imgs_path = os.path.join(image_path_frame, filename)
        img = cv2.imread(one_imgs_path)
        seq_height, seq_width = img.shape[:2]

        label_fpath = osp.join(
            seq_label_root, filename.replace("jpg", "txt").replace("png", "txt")
        )

        lines = []

        tid_curr += 1

        trans = text_char

        w = seq_width - 1
        h = seq_height - 1
        x = seq_width / 2
        y = seq_height / 2
        label_str = "0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.1f} {:.1f} {:.1f} {:.1f} {}\n".format(
            tid_curr,
            x / seq_width,
            y / seq_height,
            w / seq_width,
            h / seq_height,
            0,
            0,
            0,
            w,
            h,
            trans,
        )
        lines.append(label_str)

        write_lines(label_fpath, lines)


gen_data_path(path="/mmu-ocr/weijiawu/Data/VideoText/MOTR/Synthetic_Chinese_OCR")
