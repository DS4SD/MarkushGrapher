import logging

import torch

from markushgrapher.core.common.utils import (check_max_values,
                                              normalize_bbox_format)

logger = logging.getLogger(__name__)


def split_sentence_into_words(sentence, tokenizer):
    words = tokenizer.tokenize(sentence)
    return words


def estimate_word_width(word):
    if word == "▁":
        length = 1
    else:
        length = len([c for c in word if c != "▁"])
    return length * 12


def split_bounding_box_for_words(sentence, bounding_box, tokenizer):
    # Split the sentence into words
    words = split_sentence_into_words(sentence, tokenizer)

    # Calculate the total length of the characters in the sentence
    total_characters_width = sum(estimate_word_width(word) for word in words)

    # Get the starting position of the bounding box
    x_min, y_min, x_max, y_max = bounding_box
    current_left = x_min

    # Create a list to store individual word bounding boxes
    word_bounding_boxes = []

    for word in words:
        word_width = estimate_word_width(word)
        fraction_of_total_width = word_width / total_characters_width
        adjusted_width = (x_max - x_min) * fraction_of_total_width
        word_box = (current_left, y_min, current_left + adjusted_width, y_max)
        word_bounding_boxes.append(word_box)

        # Update the starting position for the next word (excluding spaces)
        current_left += adjusted_width

    return words, word_bounding_boxes


def normalText(t):
    if type(t) is float:
        if t == int(t):
            t = int(t)
    t = str(t)
    return t.strip()


def prepare_cells_to_text(
    cells, tokenizer, w, h, normalize_bbox, max_sequence_length=512
):
    words, bboxes = [], []
    token_idx = 0
    for cell in cells:
        if cell["text"].isspace():
            continue

        cell_bbox = [
            cell["bbox"][0] * w,
            cell["bbox"][1] * h,
            cell["bbox"][2] * w,
            cell["bbox"][3] * h,
        ]

        cell_text = cell["text"]
        split_words, split_word_bboxes = split_bounding_box_for_words(
            cell_text, cell_bbox, tokenizer
        )

        for word_text, word_bbox in zip(split_words, split_word_bboxes):
            if word_text.isspace():
                continue

            if not (normalize_bbox):
                # Map from 0-512 to 0-500 (used in trainings before 2025)
                word_bbox = normalize_bbox_format(word_bbox, w, h)

            if check_max_values(word_bbox):
                logger.debug("Cell box incorrect dimension %s", word_bbox)
                continue
            words.append(normalText(word_text))
            bboxes.append(word_bbox)
            token_idx += len(tokenizer.tokenize(normalText(word_text)))

            # TODO Find the longest prompt to remove the condition bellow
            if token_idx >= max_sequence_length - 15:
                break

        if token_idx >= max_sequence_length:
            break

        assert len(words) == len(bboxes), f"text bbox length mismatch"

    return words, bboxes, token_idx
