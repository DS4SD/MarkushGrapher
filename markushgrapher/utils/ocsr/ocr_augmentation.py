import random

import numpy as np


class OCRAugmentator:
    def __init__(self):
        self.parameters = {
            # Bounding boxes
            "bbox_position_augmentation_proportion": 0.2,
            "bbox_position_augmentation_delta_range": [-0.004, 0.004],
            "bbox_position_augmentation_size_range": [-0.002, 0.002],
            # Text
            "min_nb_characters_augmentation": 5,
            "text_augmentation_proportion": 0.005,
            "nb_text_augmentations_range": [1, 5],
            "character_substitution_proportion": 0.4,
            "character_insertion_proportion": 0.15,
            "character_deletion_proportion": 0.15,
            "character_transposition_proportion": 0.15,
            "case_alterations_proportion": 0.15,
        }
        self.strategies = [
            self.character_substitution,
            self.character_insertion,
            self.character_deletion,
            self.character_transposition,
            self.case_alterations,
        ]
        self.substitutions = {
            "0": "O",
            "O": "0",
            "1": "l",
            "l": "1",
            "5": "S",
            "S": "5",
            "8": "B",
            "B": "8",
        }

    def augment_bbox(self, bbox):
        if random.random() < self.parameters["bbox_position_augmentation_proportion"]:
            delta_x = random.uniform(
                self.parameters["bbox_position_augmentation_delta_range"][0],
                self.parameters["bbox_position_augmentation_delta_range"][1],
            )
            delta_y = random.uniform(
                self.parameters["bbox_position_augmentation_delta_range"][0],
                self.parameters["bbox_position_augmentation_delta_range"][1],
            )
            size_shift_x = random.uniform(
                self.parameters["bbox_position_augmentation_size_range"][0],
                self.parameters["bbox_position_augmentation_size_range"][1],
            )
            size_shift_y = random.uniform(
                self.parameters["bbox_position_augmentation_size_range"][0],
                self.parameters["bbox_position_augmentation_size_range"][1],
            )
            bbox = [
                bbox[0] + delta_x,
                bbox[1] + delta_y,
                bbox[2] + delta_x + size_shift_x,
                bbox[3] + delta_y + size_shift_y,
            ]
        return bbox

    def augment_text(self, text):
        if random.random() < (1 - self.parameters["text_augmentation_proportion"]):
            return text

        if len(text) > self.parameters["min_nb_characters_augmentation"]:
            nb_strategies = random.choice(
                self.parameters["nb_text_augmentations_range"]
            )
        else:
            nb_strategies = 1

        strategy_list = []
        for _ in range(nb_strategies):
            strategy_list.append(
                np.random.choice(
                    self.strategies,
                    p=[
                        self.parameters["character_substitution_proportion"],
                        self.parameters["character_insertion_proportion"],
                        self.parameters["character_deletion_proportion"],
                        self.parameters["character_transposition_proportion"],
                        self.parameters["case_alterations_proportion"],
                    ],
                )
            )
        random.shuffle(strategy_list)
        augmented_text = text
        for strategy in strategy_list:
            augmented_text = strategy(augmented_text)
        return augmented_text

    def character_substitution(self, text):
        return "".join(self.substitutions.get(c, c) for c in text)

    def character_insertion(self, text):
        insert_char = random.choice("abcdefghijklmnopqrstuvwxyz")
        pos = random.randint(0, len(text))
        return text[:pos] + insert_char + text[pos:]

    def character_deletion(self, text):
        if len(text) > 1:
            pos = random.randint(0, len(text) - 1)
            return text[:pos] + text[pos + 1 :]
        return text

    def character_transposition(self, text):
        if len(text) > 1:
            pos = random.randint(0, len(text) - 2)
            return text[:pos] + text[pos + 1] + text[pos] + text[pos + 2 :]
        return text

    def case_alterations(self, text):
        return "".join(c.lower() if random.random() < 0.5 else c.upper() for c in text)
