import logging
import random
from typing import Union

import PIL

from markushgrapher.core.common.data_preprocessing import (
    normalText, prepare_cells_to_text, split_bounding_box_for_words)
from markushgrapher.core.common.utils import (TOKEN_REGISTRY, check_max_values,
                                              normalize_bbox_format)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskCollator:
    r"""
    Collate input data as it suits each task.
    Perform masking for the self-supervised tasks.
    """

    def __init__(self, tokenizer):
        r"""
        Initialize TaskCollator
        """
        self._tokenizer = tokenizer

    def collate(
        self,
        item: dict,
        normalize_bbox: bool,
    ) -> tuple[PIL.Image, str, list[str], list[list[float], list[str]]]:
        r"""
        Parameter
        ---------
        item: A dictionary with the keys: "image", "entities", "cells", "config"

        Returns
        --------
        image: PIL image,
        instruction: The "full" prompt. It also contains the entity label
        input_ids: tokens (tokenized cell text)
        bbox_list: normalized estimated bboxes for each token
        labels: [ label from enumerations,
                  loc_{bbox_x1}, loc_{bbox_y1}, loc_{bbox_x2}, loc_{bbox_y2},
                  </s>
                ]
        """
        image = item["image"]
        w, h = image.size
        entity = item["entities"]
        cells = item["cells"]

        # Page input
        labels = []

        entity_q = entity["question"]
        entity_a = entity["answer"]

        words, bboxes, token_idx = prepare_cells_to_text(
            cells, self._tokenizer, w, h, normalize_bbox
        )
        # words:
        # ['▁where', 'in', '▁R', '1', '▁represents', '▁', 'a', '▁(', 'C', '1', '-', 'C', '6)', 'al', 'ky', 'l', '▁group', ',', '▁R', '6', '▁and', '▁R', '7', '▁may', '▁be', '▁the', '▁same', '▁or', '▁different', ',',
        # '▁and', '▁each', '▁represent', '▁', 'a', '▁hydrogen', '▁', 'atom', ',', '▁', 'a', '▁', 'hal', 'ogen', '▁', 'atom', '▁or', '▁', 'a', '▁(', 'C', '1', '-', 'C', '6)', '▁R', '2', '▁and', '▁R', '4', '▁may', '
        # ▁be', '▁the', '▁same', '▁or', '▁different', ',', '▁and', '▁each', '▁represent', '▁', 'a', '▁', 'hal', 'o', '▁(', 'C', '1', '-', 'C', '6)', '▁al', 'ky', 'l', '▁group', ',', '▁', 'm', '▁represents', '▁0,',
        # '▁1', '▁or', '▁2,', '▁and', '▁al', 'ky', 'l', '▁group', ',', '▁or', '▁', 'a', '▁salt', '▁thereof', '▁to', '▁plants', '▁or', '▁soil', '.', '▁R', '2', '▁R', '1', '▁S', '▁O', '▁', 'm', '▁O', '▁N', '▁R', '7',
        # '▁R', '6', '▁R', '4']

        # tokenizer.convert_ids_to_tokens(input_ids) (subset):
        # ['▁where', '▁in', '▁R', '▁1', '▁represents', '▁', 'a', '▁(', '▁C', '▁1', '▁', '-', '▁C', '▁', '6)', '▁al', '▁', 'ky', '▁', 'l', '▁group', '▁', ',', '▁R', '▁6', '▁and', '▁R', '▁7', '▁may', '▁be', '▁the', '▁same',
        # '▁or', '▁different', '▁', ',', '▁and', '▁each', '▁represent', '▁', 'a', '▁hydrogen', '▁', 'atom', '▁', ',', '▁', 'a', '▁', 'hal', '▁', 'ogen', '▁', 'atom', '▁or', '▁', 'a', '▁(', '▁C', '▁1', '▁', '-', '▁C',
        # '▁', '6)', '▁R', '▁2', '▁and', '▁R', '▁4', '▁may', '▁be', '▁the', '▁same', '▁or', '▁different', '▁', ',', '▁and', '▁each', '▁represent', '▁', 'a', '▁', 'hal', '▁', 'o', '▁(', '▁C', '▁1', '▁', '-', '▁C', '▁', '6)',
        # '▁al', '▁', 'ky', '▁', 'l', '▁group', '▁', ',', '▁', 'm', '▁represents', '▁0,', '▁1', '▁or', '▁2,', '▁and', '▁al', '▁', 'ky', '▁', 'l', '▁group', '▁', ',', '▁or', '▁', 'a', '▁salt', '▁thereof', '▁to', '▁plants',
        # '▁or', '▁soil', '▁', '.', '▁R', '▁2', '▁R', '▁1', '▁S','▁O', '▁', 'm', '▁O', '▁N', '▁R', '▁7', '▁R', '▁6', '▁R', '▁4']

        # ocr annotation
        # 'wherein'
        # 'R1 represents a (C1-C6)alkyl group,'
        # 'R2 and R4 may be the same or different, and each '
        #         'represent a halo (C1-C6) alkyl group,'
        # 'm represents 0, 1 or 2, and'
        # 'R6 and R7 may be the same or different, and each '
        #         'represent a hydrogen atom, a halogen atom or a (C1-C6)'
        # 'alkyl group, or a salt thereof to plants or soil.'
        # 'R2'
        # 'R1'
        # 'S'
        # 'O'
        # 'N'
        # 'R7'
        # 'R6'
        # 'R4'
        # 'O'
        # 'm'

        instruction = f"Question Answering. {entity_q}"

        labels.append(normalText(entity_a))
        labels.append("</s>")

        if normalize_bbox:
            bboxes = [
                [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h] for bbox in bboxes
            ]

        return image, instruction, words, bboxes, labels
