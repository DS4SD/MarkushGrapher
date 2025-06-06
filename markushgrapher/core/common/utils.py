import logging
import math
import os
import re
import warnings
from datetime import datetime
from typing import List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)
PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

TOKEN_REGISTRY = {
    "ecel": "<other_0>",
    "fcel": "<other_1>",
    "lcel": "<other_2>",
    "ucel": "<other_3>",
    "nl": "<other_4>",
    "xcel": "<other_125>",
    "<tr>": "<other_5>",
    "<td>": "<other_6>",
    "</td>": "<other_7>",
    "</tr>": "<other_8>",
    "<td": "<other_9>",
    ' colspan="5"': "<other_10>",
    ">": "<other_11>",
    ' colspan="3"': "<other_12>",
    ' colspan="2"': "<other_13>",
    ' colspan="4"': "<other_14>",
    ' rowspan="2"': "<other_15>",
    ' colspan="6"': "<other_16>",
    ' colspan="7"': "<other_17>",
    ' colspan="9"': "<other_18>",
    ' colspan="13"': "<other_19>",
    ' colspan="8"': "<other_20>",
    ' colspan="10"': "<other_21>",
    ' colspan="12"': "<other_22>",
    ' colspan="11"': "<other_23>",
    ' colspan="15"': "<other_24>",
    ' colspan="16"': "<other_25>",
    ' rowspan="4"': "<other_26>",
    ' rowspan="6"': "<other_27>",
    ' colspan="14"': "<other_28>",
    ' rowspan="3"': "<other_29>",
    ' rowspan="7"': "<other_30>",
    ' rowspan="10"': "<other_31>",
    ' rowspan="5"': "<other_32>",
    ' rowspan="8"': "<other_33>",
    ' rowspan="9"': "<other_34>",
    ' rowspan="11"': "<other_35>",
    ' rowspan="12"': "<other_36>",
    ' rowspan="13"': "<other_37>",
    ' rowspan="14"': "<other_38>",
    ' rowspan="15"': "<other_39>",
    ' rowspan="16"': "<other_40>",
    ' rowspan="17"': "<other_41>",
    ' rowspan="18"': "<other_42>",
    ' rowspan="19"': "<other_43>",
    ' rowspan="20"': "<other_44>",
    ' colspan="17"': "<other_45>",
    ' colspan="18"': "<other_46>",
    ' colspan="19"': "<other_47>",
    ' colspan="20"': "<other_126>",
    ' rowspan="21"': "<other_48>",
    ' rowspan="22"': "<other_49>",
    ' rowspan="23"': "<other_50>",
    ' rowspan="24"': "<other_51>",
    ' rowspan="25"': "<other_52>",
    ' rowspan="26"': "<other_53>",
    ' rowspan="27"': "<other_54>",
    ' rowspan="28"': "<other_55>",
    ' rowspan="29"': "<other_56>",
    ' rowspan="30"': "<other_57>",
    ' rowspan="31"': "<other_58>",
    ' rowspan="32"': "<other_59>",
    ' colspan="21"': "<other_60>",
    ' colspan="22"': "<other_61>",
    ' colspan="23"': "<other_62>",
    ' colspan="24"': "<other_63>",
    ' colspan="25"': "<other_64>",
    ' colspan="26"': "<other_65>",
    ' colspan="27"': "<other_66>",
    ' colspan="28"': "<other_67>",
    ' colspan="29"': "<other_68>",
    ' colspan="30"': "<other_69>",
    ' colspan="31"': "<other_70>",
    ' colspan="32"': "<other_71>",
    ' colspan="33"': "<other_72>",
    ' colspan="34"': "<other_73>",
    ' colspan="35"': "<other_74>",
    ' colspan="36"': "<other_75>",
    ' colspan="37"': "<other_76>",
    ' colspan="38"': "<other_77>",
    ' colspan="39"': "<other_78>",
    ' colspan="40"': "<other_79>",
    ' rowspan="33"': "<other_80>",
    ' rowspan="34"': "<other_81>",
    ' rowspan="35"': "<other_82>",
    ' rowspan="36"': "<other_83>",
    ' rowspan="37"': "<other_84>",
    ' rowspan="38"': "<other_85>",
    ' rowspan="39"': "<other_86>",
    ' rowspan="40"': "<other_87>",
    ' rowspan="41"': "<other_88>",
    ' rowspan="42"': "<other_89>",
    ' rowspan="43"': "<other_90>",
    ' rowspan="44"': "<other_91>",
    ' rowspan="45"': "<other_92>",
    ' rowspan="46"': "<other_93>",
    ' rowspan="47"': "<other_94>",
    ' rowspan="48"': "<other_95>",
    ' rowspan="49"': "<other_96>",
    ' rowspan="50"': "<other_98>",
    ' rowspan="51"': "<other_99>",
    ' rowspan="52"': "<other_100>",
    ' rowspan="53"': "<other_101>",
    ' rowspan="54"': "<other_102>",
    ' rowspan="55"': "<other_103>",
    ' colspan="41"': "<other_104>",
    ' colspan="42"': "<other_105>",
    ' colspan="43"': "<other_106>",
    ' colspan="44"': "<other_107>",
    ' colspan="45"': "<other_108>",
    ' colspan="46"': "<other_109>",
    ' colspan="47"': "<other_110>",
    ' colspan="48"': "<other_111>",
    ' colspan="49"': "<other_112>",
    ' colspan="50"': "<other_113>",
    ' colspan="51"': "<other_114>",
    ' colspan="52"': "<other_115>",
    ' colspan="53"': "<other_116>",
    ' colspan="54"': "<other_117>",
    ' colspan="55"': "<other_118>",
    "</tbody>": "<other_119>",
    "</thead>": "<other_120>",
    "<tbody>": "<other_121>",
    "<thead>": "<other_122>",
    "<key>": "<other_123>",
    "<value>": "<other_124>",
}

TOKEN_REGISTRY_OP = {v: k for k, v in TOKEN_REGISTRY.items()}


def calculate_iou(box1, box2):
    # Parse x1, y1, x2, y2 from location tokens
    x1_1, y1_1, x2_1, y2_1 = [int(box1[i][5:-1]) for i in range(4)]
    x1_2, y1_2, x2_2, y2_2 = [int(box2[i][5:-1]) for i in range(4)]

    # Calculate intersection area
    intersection_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(
        0, min(y2_1, y2_2) - max(y1_1, y1_2)
    )

    # Calculate union area
    box1_area = np.abs(x2_1 - x1_1) * np.abs(y2_1 - y1_1)
    box2_area = np.abs(x2_2 - x1_2) * np.abs(y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    # Calculate IOU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(
        folder,
        max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])),
    )


def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)


def normalize_bbox(bbox, size, scale=1000):
    return [
        int(clamp((scale * bbox[0] / size[0]), 0, scale)),
        int(clamp((scale * bbox[1] / size[1]), 0, scale)),
        int(clamp((scale * bbox[2] / size[0]), 0, scale)),
        int(clamp((scale * bbox[3] / size[1]), 0, scale)),
    ]


def normalText(t):
    if type(t) is float:
        if t == int(t):
            t = int(t)
    t = str(t)
    return t.strip()


def check_max_values(bounding_box, max_value=500):
    return any(coord > max_value for coord in bounding_box)


def normalize_bbox_format(bbox, image_width, image_height):
    xmin, ymin, xmax, ymax = bbox
    normalized_xmin = int((xmin / image_width) * 500)
    normalized_ymin = int((ymin / image_height) * 500)
    normalized_xmax = int((xmax / image_width) * 500)
    normalized_ymax = int((ymax / image_height) * 500)
    return normalized_xmin, normalized_ymin, normalized_xmax, normalized_ymax
