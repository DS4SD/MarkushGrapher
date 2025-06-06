import logging
import random
import re
from pprint import pprint

import numpy as np
from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageEnhance
from torch.utils.data import Dataset

from markushgrapher.core.common.markush_tokenizer import MarkushTokenizer
from markushgrapher.utils.common import encode_item
from markushgrapher.utils.ocsr.definition_group_selector import \
    DefinitionGroupSelector
from markushgrapher.utils.ocsr.image_augmentation import get_transforms_dict
from markushgrapher.utils.ocsr.ocr_augmentation import OCRAugmentator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MDU_Dataset(Dataset):
    def __init__(
        self, config, data_args, tokenizer, processor, collator, split="train"
    ):
        r""" """
        logger.info("Building Dataset: %s for %s", config["name"], config["task"])

        self._config = config
        self._data_args = data_args

        if split == "val":
            split = "test"
        self._split = split

        dataset_dir = config["dataset_path"]
        dataset_dict = load_from_disk(dataset_dir, keep_in_memory=False)
        self._ds = dataset_dict[split]
        print(self._ds)

        if "test" in dataset_dict:
            print(f"Test size: {len(dataset_dict['test'])}")
        if "train" in dataset_dict:
            print(f"Train size: {len(dataset_dict['train'])}")

        self._max_sequence_length = data_args.max_seq_length
        self._collator = collator
        self._tokenizer = tokenizer

        self.markush_tokenizer = MarkushTokenizer(
            self._tokenizer,
            dataset_path=self._config["dataset_path"],
            encode_position=self._config["encode_position"],
            grounded_smiles=self._config["grounded_smiles"],
            encode_index=self._config["encode_index"],
            training_dataset_name=self._config["training_dataset_name"],
            condense_labels=self._config["condense_labels"],
        )
        self._processor = processor["no_ocr"]
        self._tokenize_unit = "word"
        self.transforms_dict_train = get_transforms_dict(augment=True)
        self.transforms_dict_predict = get_transforms_dict(augment=False)
        self.definition_group_selector = DefinitionGroupSelector(tokenizer)
        self.ocr_augmentator = OCRAugmentator()

    def __len__(self):
        return len(self._ds)

    def replace_cxsmiles_with_cxsmiles_opt(self, string, replacement):
        pattern = r"(<cxsmi>)(.*?)(</cxsmi>)"

        def replacer(match):
            return f"{match.group(1)}{replacement}{match.group(3)}"

        new_string = re.sub(pattern, replacer, string, count=1)
        return new_string

    def order_cells(self, cells):
        cells = sorted(cells, key=lambda d: (d["bbox"][1], d["bbox"][0]))
        return cells

    def crop_image_and_update_bboxes(self, page_image, cells, bbox_crop):
        new_page_image = page_image.crop(bbox_crop)
        new_cells = []
        for cell in cells:
            new_cell = {
                "text": cell["text"],
                "bbox": [
                    (cell["bbox"][0] * page_image.width - bbox_crop[0])
                    / new_page_image.width,
                    (cell["bbox"][1] * page_image.height - bbox_crop[1])
                    / new_page_image.height,
                    (cell["bbox"][2] * page_image.width - bbox_crop[0])
                    / new_page_image.width,
                    (cell["bbox"][3] * page_image.height - bbox_crop[1])
                    / new_page_image.height,
                ],
            }
            new_cells.append(new_cell)
        return new_page_image, new_cells

    def __getitem__(self, idx: int, verbose=False):
        row = self._ds[idx]

        # Create annotation
        if not (self._config["encode_position"]):
            cxsmiles_star_raw = row["cxsmiles_opt"] + "![[0,0]]"
        else:
            pass
        answer = row["annotation"]
        if verbose:
            print(f"Answer: {answer}")
        answer = self.replace_cxsmiles_with_cxsmiles_opt(answer, cxsmiles_star_raw)
        # print(answer)

        # answer = "<markush><cxsmi>C<r>R1</r>.C1C(CCN1)OC |m:0:2.3.4.5.6,Sg:n:7:n:ht|![[0,0]]</cxsmi><stable>U3:H2As<ns>B6:an alkyl group<ns>R:optional<n>sulfur<n>oxygen<n>as defined in claim 1<ns>V41:a bromine atom<ns>Rq:C20H41<n>C24H49<ns>D30:N<ns>E:N<n>CH</stable></markush>"
        if verbose:
            print(f"Answer  2: {answer}")

        # 01.01 (and 30.10)
        # page_image = row["page_image"]

        # 25.10 (and 08.11)
        page_image = row["page_image"].resize((512, 512), resample=Image.LANCZOS)

        entities_row = {
            "question": "What markush structure is in the image?",
            "answer": answer,
            "bbox": [0, 0, page_image.size[0], page_image.size[1]],
        }

        # Augment
        if (self._split == "train") or (
            (self._config["augment_test"] and self._split == "test")
        ):  # and (not("lum_test" in self._config["dataset_path"])) or not("uspto_markush" in self._config["dataset_path"]))):
            # Augment image
            image = np.array(page_image, dtype=np.float32)
            bboxes = []
            for cell in row["cells"]:
                bbox = cell["bbox"]
                if (bbox == [0, 0, 0, 0]) or (bbox == [1000, 1000, 1000, 1000]):
                    continue
                # Map bboxes between [0, 1] to [0, image size] and clip bboxes
                # TODO Extreme bboxes positions should be debugged
                bbox = [
                    min(
                        max(bbox[0] * page_image.size[0], 0),
                        float(page_image.size[0] - 1),
                    ),
                    min(
                        max(bbox[1] * page_image.size[1], 0),
                        float(page_image.size[1] - 1),
                    ),
                    max(
                        min(
                            bbox[2] * page_image.size[0], float(page_image.size[0] - 1)
                        ),
                        0,
                    ),
                    max(
                        min(
                            bbox[3] * page_image.size[1], float(page_image.size[1] - 1)
                        ),
                        0,
                    ),
                ]
                bbox = [
                    bbox[0],
                    bbox[1],
                    max(bbox[2] - bbox[0], 0.1),
                    max(bbox[3] - bbox[1], 0.1),
                ] + ["fill"]
                if verbose:
                    print("bbox:", bbox)
                bboxes.append(bbox)
            transformed = self.transforms_dict_train["standard"](
                image=image, bboxes=bboxes
            )
            image = transformed["image"]
            image = Image.fromarray(np.uint8(image)).convert(
                "RGB"
            )  # Does this convert alter image quality?
            transformed_bboxes = transformed["bboxes"]
            new_cells = []
            for transformed_bbox, cell in zip(transformed_bboxes, row["cells"]):
                if verbose:
                    print("transformed_bbox:", transformed_bbox)
                bbox = [
                    transformed_bbox[:4][0] / image.size[0],
                    transformed_bbox[:4][1] / image.size[1],
                    transformed_bbox[:4][2] / image.size[0],
                    transformed_bbox[:4][3] / image.size[1],
                ]  # Map bboxes between [0, 512] (image size) to [0, 1]
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                new_cells.append({"bbox": bbox, "text": cell["text"]})
            row["cells"] = new_cells

            # Augment OCR text
            new_cells = []
            for cell in row["cells"]:
                augmented_text = self.ocr_augmentator.augment_text(cell["text"])
                new_cells.append({"bbox": cell["bbox"], "text": augmented_text})
            row["cells"] = new_cells

            # Augment OCR boxes positions
            new_cells = []
            for cell in row["cells"]:
                augmented_bbox = self.ocr_augmentator.augment_bbox(cell["bbox"])
                new_cells.append({"bbox": augmented_bbox, "text": cell["text"]})
            row["cells"] = new_cells
        else:
            # 25.10 (and 08.11)
            image = page_image

            # 30.10
            # image = np.array(page_image, dtype=np.float32)
            # transformed = self.transforms_dict_predict["standard"](image=image, bboxes=[])
            # image = transformed["image"]
            # image = Image.fromarray(np.uint8(image)).convert('RGB') # TODO check if this conversion has impact on image quality.
            # TODO check how image quality can be improved during inference.

        # Order cells
        # Note: Boxes that should be ordered from left to right are the splitted boxes, after encode_item() and collate().
        # As the splitting code order splitted boxes from left to right, it should be correctly done.
        row["cells"] = self.order_cells(row["cells"])

        item = {
            "image": image,
            "entities": entities_row,
            "cells": row["cells"],
            "config": self._config,
        }
        if verbose:
            draw = ImageDraw.Draw(image)
            for cell in item["cells"]:
                bbox = cell["bbox"]
                if (bbox == [0, 0, 0, 0]) or (bbox == [1000, 1000, 1000, 1000]):
                    continue
                bbox = [p for p in bbox]
                draw.rectangle(
                    (
                        (bbox[0] * page_image.size[0], bbox[1] * page_image.size[1]),
                        (bbox[2] * page_image.size[0], bbox[3] * page_image.size[1]),
                    ),
                    outline="black",
                    width=5,
                )
            image.show()

        logger.debug("--: %s", self._config["task"])

        return encode_item(
            item,
            self._processor,
            self._tokenizer,
            self.markush_tokenizer,
            self._collator,
            self._split,
            self.definition_group_selector,
            self._config["encode_definition_group"],
        )
