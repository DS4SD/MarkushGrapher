#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re


class DefinitionGroupSelector:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rgroup_vocabulary = {
            "rlabel": [
                "A",
                "B",
                "D",
                "E",
                "G",
                "J",
                "K",
                "L",
                "M",
                "Q",
                "T",
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
            ],
            "rlabel_number": [""]
            + [str(i) for i in list(range(0, 10))]
            + ["0" + str(i) for i in list(range(0, 10))],
            "rlabel_ring": [
                "A",
                "E",
                "G",
                "J",
                "K",
                "L",
                "M",
                "Q",
                "R",
                "T",
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
            ],
            "rlabel_ring_number": [""]
            + [str(i) for i in list(range(0, 100))]
            + ["0" + str(i) for i in list(range(0, 100))],
        }

    def detect_markush_structure_image_tokens(
        self,
        bboxes,
        input_tokens_decoded,
        horizontal_threshold=15,
        vertical_threshold=10,
    ):
        def are_close(box1, box2, horizontal_threshold, vertical_threshold):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            horizontal_close = (
                max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) > 0
                or abs(x1_min - x2_max) < horizontal_threshold
                or abs(x2_min - x1_max) < horizontal_threshold
            )
            vertical_close = (
                max(0, min(y1_max, y2_max) - max(y1_min, y2_min)) > 0
                or abs(y1_min - y2_max) <= vertical_threshold
                or abs(y2_min - y1_max) <= vertical_threshold
            )
            return horizontal_close and vertical_close

        groups = []
        for i, bbox in enumerate(bboxes):
            merged = False
            for group in groups:
                # Check if this bbox is close to any bbox in the current group
                if any(
                    are_close(
                        bbox,
                        bboxes[existing_box],
                        horizontal_threshold,
                        vertical_threshold,
                    )
                    for existing_box in group
                ):
                    group.append(i)
                    merged = True
                    break
            if not merged:
                groups.append([i])

        # Concatenate clusters of less than 5 indices
        isolated_boxes_indices = []
        for group in groups:
            if len(group) > 5:
                continue
            isolated_boxes_indices.extend(group)
        return isolated_boxes_indices

    def detect_r_labels_positions(self, subwords, source=None):
        """
        image input: ['▁O', '▁', 'Y', '▁3', '▁P', '▁L', '▁5', '▁', 'Y']
        output: {'Y3': [2, 3], 'L5': [5, 6], "Y":[7]}
        """
        r_labels_positions = []
        r_labels_positions_dict = {}

        # Clean subwords by removing '▁' characters for matching purposes, but preserve their positions
        cleaned_subwords = [subword.replace("▁", "") for subword in subwords]
        concatenated = "".join(cleaned_subwords)

        matches = re.finditer(
            r"([ABDEGJKLMQRTUVWXYZ]\d{1,2})|([ABDEGJKLMQRTUVWXYZ])(?!\d)", concatenated
        )
        cumulative_length = 0
        subword_indices = []
        for _, subword in enumerate(cleaned_subwords):
            start_idx = cumulative_length
            end_idx = cumulative_length + len(subword)
            subword_indices.append((start_idx, end_idx))
            cumulative_length = end_idx

        for match_ in matches:
            variable = match_.group()
            match_start = match_.start()
            match_end = match_.end()
            matched_subword_indices = []
            for idx, (start_idx, end_idx) in enumerate(subword_indices):
                if match_start < end_idx and match_end > start_idx:
                    matched_subword_indices.append(idx)
            r_labels_positions.append([variable, matched_subword_indices])
            r_labels_positions_dict[variable] = matched_subword_indices
        return r_labels_positions, r_labels_positions_dict

    def select(self, input_ids, boxes, verbose=False):
        """Create R-group definitions. Definitions are splits of the text description based on R-labels occurences."""
        input_tokens_decoded = self.tokenizer.convert_ids_to_tokens(input_ids)
        if verbose:
            print("input_tokens_decoded:", input_tokens_decoded)

        # Remove tokens from the question (and last </s> token)
        end_question_index = float("inf")
        for i, input_token_decoded in enumerate(input_tokens_decoded):
            # Split after first occurence of "</s>"
            if input_token_decoded == "</s>":
                end_question_index = i
                break
        input_tokens_decoded = input_tokens_decoded[end_question_index + 1 : -1]
        boxes = boxes[end_question_index + 1 : -1]

        # Isolate tokens from the Markush structure image
        isolated_boxes_indices = self.detect_markush_structure_image_tokens(
            boxes.tolist(), input_tokens_decoded
        )

        # Get r_labels_positions from image
        max_index = 0
        if len(isolated_boxes_indices) > 0:
            max_index = max(isolated_boxes_indices)
        input_tokens_decoded_image = input_tokens_decoded[: max_index + 1]
        _, r_labels_positions_image_dict = self.detect_r_labels_positions(
            input_tokens_decoded_image, source="image"
        )
        if verbose:
            print("r_labels_positions_image_dict:", r_labels_positions_image_dict)

        # Get r_labels_positions from text
        max_index = 0
        if len(isolated_boxes_indices) > 0:
            max_index = max(isolated_boxes_indices)
        input_tokens_decoded_text = input_tokens_decoded[max_index + 1 :]
        r_labels_positions_text, r_labels_positions_text_dict = (
            self.detect_r_labels_positions(input_tokens_decoded_text, source="text")
        )
        if verbose:
            print("r_labels_positions_text_dict:", r_labels_positions_text_dict)

        # Filter detected r_labels in text
        """
        Example
        Input: ['▁Question', '▁Answer', 'ing', '.', '▁What', '▁mark', 'ush', '▁structure', '▁is', '▁in', '▁the', '▁image', '?', '</s>', '▁O', '▁', \
            'Y', '▁3', '▁P', '▁L', '▁5', '▁', 'Y', '▁L', '▁5', '▁is', '▁independently', '▁selected', '▁from', '▁', 'CC', '▁(', '▁C', '▁', ')', '▁=', \
            '▁S', '▁', ',', '▁CSS', '▁C', '▁', ',', '▁C', '▁SC', '▁(', '▁C', '▁', ')', '▁=', '▁O', '▁', ',', '▁and', '▁', 'CS', '▁(', '▁C', '▁', ')', \
            '▁(', '▁=', '▁', '-', '▁O', '▁', ')', '▁=', '▁O', '▁', ';', '▁', 'Y', '▁3', '▁is', '▁Ox', '▁', 'y', '▁', 'gen', '▁', ',', '▁Ber', '▁', \
            'y', '▁', 'll', '▁', 'ium', '▁', ',', '▁St', '▁', 'ront', '▁', 'ium', '▁', ',', '▁or', '▁Calcium', '▁', '.', '</s>']
        r_labels_positions_image_dict: {'Y3': [2, 3], 'L5': [5, 6], 'Y': [8]}
        r_labels_positions_text_dict: {'L5': [0, 1], 'Y3': [49, 50], 'B': [59]}
        """
        remove_text_indices = []
        remove_labels = []
        for i in range(len(r_labels_positions_text)):
            if not (r_labels_positions_text[i][0] in r_labels_positions_image_dict):
                remove_text_indices.append(i)
                remove_labels.append(r_labels_positions_text[i][0])
        r_labels_positions_text_dict = {
            k: v
            for k, v in r_labels_positions_text_dict.items()
            if not (k in remove_labels)
        }
        r_labels_positions_text = [
            v
            for i, v in enumerate(r_labels_positions_text)
            if not (i in remove_text_indices)
        ]
        if verbose:
            print("r_labels_positions_text:", r_labels_positions_text)
            print("r_labels_positions_text_dict:", r_labels_positions_text_dict)

        definition_groups = []
        text_offset = end_question_index + max_index + 1
        image_offset = end_question_index + 1
        for i in range(len(r_labels_positions_text)):
            # For definition text section, span all tokens in between the r_labels
            if (i + 1) == len(r_labels_positions_text):
                text_span = [
                    r_labels_positions_text[i][1][0] + text_offset,
                    len(input_tokens_decoded) + text_offset,
                ]
            else:
                text_span = [
                    r_labels_positions_text[i][1][0] + text_offset,
                    r_labels_positions_text[i + 1][1][0] + text_offset,
                ]

            if len(r_labels_positions_image_dict[r_labels_positions_text[i][0]]) > 1:
                end_image_span = r_labels_positions_image_dict[
                    r_labels_positions_text[i][0]
                ][1]
            else:
                end_image_span = r_labels_positions_image_dict[
                    r_labels_positions_text[i][0]
                ][0]
            image_span = [
                r_labels_positions_image_dict[r_labels_positions_text[i][0]][0]
                + image_offset,
                end_image_span + image_offset,
            ]
            definition_groups.append(image_span + text_span)

        if verbose:
            print("definition_groups:", definition_groups)
        return definition_groups
