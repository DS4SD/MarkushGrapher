#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from textwrap import wrap

from PIL import Image, ImageDraw
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
import os
from pprint import pprint

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import ImageFont

from markushgrapher.utils.ocsr.utils_markush import (canonicalize_markush,
                                                     display_markush)


def pad_lists(lists):
    max_len = max(len(sublist) for sublist in lists)
    padded_lists = [sublist + [""] * (max_len - len(sublist)) for sublist in lists]
    return padded_lists


def adjust_table(table, stable_data, table_fontsize):
    for (i, j), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("gray")
        if j == 0:
            cell.set_facecolor("whitesmoke")
            cell.set_text_props(color="black")
            cell.set_text_props(fontstyle="italic")

    table.auto_set_column_width(col=list(range(len(stable_data[0]))))
    table.auto_set_font_size(False)
    table.set_fontsize(table_fontsize)
    return table


def split_words_by_length(words, max_length=100):
    sublists = []
    current_sublist = [words[0]]
    current_length = len(words[0])
    for word in words[1:]:
        word_length = len(word)
        if current_length + word_length <= max_length:
            current_sublist.append(word)
            current_length += word_length
        else:
            sublists.append(current_sublist)
            current_sublist = [word]
            current_length = word_length
    if current_sublist:
        sublists.append(current_sublist)
    return sublists


def display_eval_sample(
    image,
    encoding_bbox,
    input_ids,
    input_text,
    label_text,
    predicted_text,
    gt_smiles,
    gt_smiles_opt,
    predicted_smiles,
    predicted_smiles_opt,
    gt_stable,
    predicted_stable,
    dataset_config,
    output_path,
    tokenizer,
    display_errors=True,
    display_markush_evaluation=False,
    max_character_per_line=120,
    debug=False,
    text_fontsize=15,
    table_fontsize=13,
    title_fontsize=15,
    max_table_width_characters=45,
    display_ocr_cells=True,
):

    fig = plt.figure(figsize=(30, 30))  # 17, 17
    gs = gridspec.GridSpec(4, 4)
    ax1 = fig.add_subplot(gs[:2, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, 2:])
    ax4 = fig.add_subplot(gs[2, 2:])
    ax5 = fig.add_subplot(gs[3, 2:])
    ax6 = fig.add_subplot(gs[2:, :2])
    plt.subplots_adjust(wspace=0, hspace=0)

    # Overlap input image with text cells
    image = image.convert("RGBA")
    if display_ocr_cells:
        boxes_image = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(boxes_image)
        rescale_factor = image.size[0] / 500
        font = ImageFont.FreeTypeFont(
            os.path.dirname(__file__) + "/../../../data/fonts/calibri.ttf", size=20
        )

        # Note: OCR text tokens and boxes are not obtained using only 'tokenizer.tokenize(input_text)'.
        # The tokenization is defined in task_collator.py, _question_answering_collator()
        if tokenizer is None:
            texts = []
        else:
            texts = [t for t in tokenizer.convert_ids_to_tokens(input_ids)]

        for bbox, text in zip(encoding_bbox[0], texts):
            bbox = bbox.tolist()
            bbox = (
                (int(bbox[0] * rescale_factor), int(bbox[1] * rescale_factor)),
                (int(bbox[2] * rescale_factor), int(bbox[3] * rescale_factor)),
            )
            draw.rectangle(
                bbox, fill=(255, 0, 0, 75), outline=(255, 0, 0, 125), width=2
            )
        for bbox, text in zip(encoding_bbox[0], texts):
            bbox = bbox.tolist()
            bbox = (
                (int(bbox[0] * rescale_factor), int(bbox[1] * rescale_factor)),
                (int(bbox[2] * rescale_factor), int(bbox[3] * rescale_factor)),
            )
            text = text.replace("▁", " ")
            # draw.text((bbox[0][0], bbox[0][1]), text, font=font, fill=(255, 0, 0, 255))

        image = Image.alpha_composite(image, boxes_image)

    # Display input image
    ax1.imshow(image)
    ax1.set_title(
        "  Input image and OCR cells",
        loc="left",
        fontweight="bold",
        fontsize=title_fontsize,
    )
    if not (debug):
        ax1.axis("off")

    # Add line between ground-truth and prediction
    line = Line2D(
        [0, 1], [0.5, 0.5], transform=fig.transFigure, color="black", linewidth=2
    )
    fig.add_artist(line)

    # Display input text and label
    display_label = (
        "Input: \n"
        + "\n".join(wrap(f"{input_text}", max_character_per_line))
        + "\n \n"
        + "Ground-truth: \n"
        + "\n".join(wrap(f"{label_text}", max_character_per_line))
        + "\n"
    )
    text = ax2.text(
        0.02,
        0.98,
        display_label,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=text_fontsize,
    )
    ax2.set_title(
        "  Input text and ground-truth",
        loc="left",
        fontweight="bold",
        fontsize=title_fontsize,
    )
    if not (debug):
        ax2.axis("off")

    # Get stable label
    if not (gt_stable is None):
        stable_data = []
        for k, v in gt_stable.items():
            sublists = split_words_by_length(v, max_length=max_table_width_characters)
            for v in sublists:
                stable_data.append([k] + v)
        if stable_data == []:
            stable_data = [" "]
        else:
            stable_data = pad_lists(stable_data)
    else:
        stable_data = [" "]
    # Display stable label
    table = ax3.table(
        cellText=stable_data, cellLoc="left", loc="upper left", colLoc="left"
    )
    ax3.set_title(
        "  Ground-truth substituent table",
        loc="left",
        fontweight="bold",
        fontsize=title_fontsize,
    )
    if not (debug):
        ax3.axis("off")
    table = adjust_table(table, stable_data, table_fontsize)

    # Display predicted text
    display_label = "\n".join(wrap(f"{predicted_text}", max_character_per_line)) + "\n"
    text = ax4.text(
        0.02,
        0.98,
        display_label,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=text_fontsize,
    )
    # text_height = (text.get_window_extent(renderer=fig.canvas.get_renderer()).height) / (fig.get_size_inches()[0] * fig.dpi)
    ax4.set_title(
        "  Prediction", loc="left", fontweight="bold", fontsize=title_fontsize
    )
    if not (debug):
        ax4.axis("off")

    # Get stable label
    if not (predicted_stable is None) and not (predicted_stable == {}):
        stable_data = []
        for k, v in predicted_stable.items():
            sublists = split_words_by_length(v, max_length=max_table_width_characters)
            for v in sublists:
                stable_data.append([k] + v)
        stable_data = pad_lists(stable_data)
    else:
        stable_data = [" "]
    # Display stable prediction
    table = ax5.table(
        cellText=stable_data, cellLoc="left", loc="upper left", colLoc="left"
    )
    ax5.set_title(
        "  Predicted substituent table",
        loc="left",
        fontweight="bold",
        fontsize=title_fontsize,
    )
    if not (debug):
        ax5.axis("off")
    table = adjust_table(table, stable_data, table_fontsize)

    # Get predicted molecule image
    failed = False
    if predicted_smiles is None:
        print("Predicted molecule can not be read")
        if display_errors:
            molecule = Chem.MolFromSmiles("")
            failed = True
        else:
            return False
    if not (failed):
        parser_params = Chem.SmilesParserParams()
        parser_params.strictCXSMILES = False
        parser_params.sanitize = False
        parser_params.removeHs = False
        molecule = Chem.MolFromSmiles(predicted_smiles, parser_params)
    if molecule is None:
        print("Predicted molecule can not be read")
        if display_errors:
            molecule = Chem.MolFromSmiles("")
            failed = True
        else:
            return False
    if (dataset_config["name"] == "ocsr") or failed:
        image = Chem.Draw.MolToImage(molecule)
    elif (dataset_config["name"] == "ocxsr") or (dataset_config["name"] == "mdu"):
        try:
            image = display_markush(predicted_smiles)
            if image is None:
                image = Chem.Draw.MolToImage(Chem.MolFromSmiles(""))

        except Exception as e:
            print(f"Markush structure display failed: {e}")
            return False

        if display_markush_evaluation:
            from markushgrapher.utils.ocsr.utils_evaluation import \
                compute_markush_prediction_quality  # Due to circular import error

            try:
                predicted_smiles = canonicalize_markush(predicted_smiles)
            except:
                predicted_smiles = ""
            try:
                gt_smiles = canonicalize_markush(gt_smiles)
                # Debugging
                # print("gt_smiles", gt_smiles)
                # print("predicted_smiles", predicted_smiles)
                scores = compute_markush_prediction_quality(
                    predicted_smiles,
                    gt_smiles,
                    remove_stereo=True,
                    remove_double_bond_stereo=True,
                )
            except Exception as e:
                print(e)
                print(
                    f"Error in compute_markush_prediction_quality (in display_eval_sample) for: gt: {gt_smiles} and prediction: {predicted_smiles}"
                )
                scores = defaultdict(int)
            display_scores = (
                f"CXSMILES equality: {scores['cxsmi_equality']} \n"
                + f"R: {scores['r']} \n"
                + f"M: {scores['m']} \n"
                + f"Sg: {scores['sg']} \n"
                + f"InChI equality: {scores['inchi_equality']}"
            )

            # Display markush evaluation
            ax6.text(0.02, 0.88, display_scores, transform=ax6.transAxes, color="red")

    # Display predicted image
    ax6.imshow(image)
    ax6.set_title(
        "  Predicted CXSMILES", loc="left", fontweight="bold", fontsize=title_fontsize
    )
    if not (debug):
        ax6.axis("off")

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(output_path)
    plt.close()
    return True
