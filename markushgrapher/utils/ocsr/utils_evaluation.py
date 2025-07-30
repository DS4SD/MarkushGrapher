#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import math
import os
import pickle
import re
import warnings
from collections import defaultdict
from pprint import pprint

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFMCS, rdmolfiles
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.inchi import MolToInchi
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
import torch
from markushgenerator.cxsmiles_tokenizer import CXSMILESTokenizer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from transformers import logging as transformers_logging

from markushgrapher.utils.ocsr.utils_display import display_eval_sample
from markushgrapher.utils.ocsr.utils_markush import (canonicalize_markush,
                                                     get_molecule_from_smiles)
from markushgrapher.utils.ocsr.utils_postprocessing import \
    MoleculePostprocessor


def get_smiles_metrics(
    model,
    dataset,
    max_eval_samples,
    tokenizer,
    training_smiles,
    markush_tokenizer,
    cxsmiles_tokenizer,
    display_eval_samples,
    device,
    display_samples_output_dir,
    training_args,
    model_args,
    config,
    selected_indices=None,
    read_predictions=True,
    overwrite_predictions=True,
    save_scores=True,
    display_markush_evaluation=False,
    display_errors=True,
    max_display_eval_samples=20,
    verbose=False,
    markush_tokenizer_training=None,
    cxsmiles_tokenizer_training=None,
    metrics_prefix="",
):

    if markush_tokenizer_training == None:
        markush_tokenizer_training = markush_tokenizer
    if cxsmiles_tokenizer_training == None:
        cxsmiles_tokenizer_training = cxsmiles_tokenizer

    metrics = {}
    predicted_smiles_list = []
    predicted_smiles_opt_list = []
    predicted_stable_list = []
    gt_smiles_list = []
    gt_smiles_opt_list = []
    gt_stable_list = []

    if read_predictions and os.path.exists(
        f"{training_args.output_dir}/{metrics_prefix}gt_smiles_list_{max_eval_samples}.pkl"
    ):
        with open(
            f"{training_args.output_dir}/{metrics_prefix}gt_smiles_list_{max_eval_samples}.pkl",
            "rb",
        ) as f:
            gt_smiles_list = pickle.load(f)
        with open(
            f"{training_args.output_dir}/{metrics_prefix}gt_smiles_opt_list_{max_eval_samples}.pkl",
            "rb",
        ) as f:
            gt_smiles_opt_list = pickle.load(f)
        with open(
            f"{training_args.output_dir}/{metrics_prefix}predicted_smiles_list_{max_eval_samples}.pkl",
            "rb",
        ) as f:
            predicted_smiles_list = pickle.load(f)
        with open(
            f"{training_args.output_dir}/{metrics_prefix}predicted_smiles_opt_list_{max_eval_samples}.pkl",
            "rb",
        ) as f:
            predicted_smiles_opt_list = pickle.load(f)
    else:
        if not (os.path.exists(display_samples_output_dir)):
            os.mkdir(display_samples_output_dir)

        print("--- Starting auto-regressive evaluation ---")
        for idx in tqdm(range(min(len(dataset), max_eval_samples))):
            encoding = dataset.__getitem__(int(idx))

            if selected_indices and not (idx in selected_indices):
                continue

            # Process and batch inputs
            encoding["input_ids"] = (
                encoding["input_ids"].type(torch.long).unsqueeze(0).to(device)
            )
            encoding["bbox"] = (
                encoding["bbox"].type(torch.float).unsqueeze(0).to(device)
            )
            encoding["attention_mask"] = (
                encoding["attention_mask"].type(torch.long).unsqueeze(0).to(device)
            )
            encoding["decoder_attention_mask"] = (
                encoding["decoder_attention_mask"]
                .type(torch.long)
                .unsqueeze(0)
                .to(device)
            )
            encoding["pixel_values"] = encoding["pixel_values"].unsqueeze(0).to(device)
            encoding["labels"] = (
                encoding["labels"].type(torch.long).unsqueeze(0).to(device)
            )

            image = encoding["image"]
            if "decoder_attention_mask" in encoding:
                del encoding["attention_mask"]
                del encoding["decoder_attention_mask"]
                del encoding["image"]

            input_text = tokenizer.decode(encoding["input_ids"][0])
            if verbose:
                print(f"input_text: {input_text}")

            if config["udop_tokenizer_only"]:
                label_text = tokenizer.decode(encoding["labels"][0])
                label_text = (
                    label_text.replace("<unk>", "<")
                    .replace("! [ [ 0, 0 ] ]", "")
                    .replace(" ", "")
                )
                print("GT:", label_text)
            else:
                label_text = markush_tokenizer.decode_plus_decode_other_tokens(
                    encoding["labels"][0]
                )
            if verbose:
                print(f"label_text: {label_text}")
            if config["name"] == "ocsr":
                gt_smiles = (
                    label_text.replace("<smi>", "")
                    .replace("</smi>", "")
                    .replace("</s>", "")
                    .replace(" ", "")
                )

            if config["name"] == "ocxsr":
                gt_smiles_opt = (
                    label_text.replace("<cxsmi>", "")
                    .replace("</cxsmi>", "")
                    .replace("</s>", "")
                    .replace(" ", "")
                )
                if verbose:
                    print(f"gt_smiles_opt: {gt_smiles_opt}")
                try:
                    gt_smiles = cxsmiles_tokenizer.convert_opt_to_out(gt_smiles_opt)
                    if verbose:
                        print(f"gt_smiles: {gt_smiles}")
                except Exception as e:
                    if verbose:
                        print(
                            f"Error {e} in cxsmiles_tokenizer.convert_opt_to_out for {gt_smiles_opt}"
                        )
                    gt_smiles = None
            if config["name"] == "mdu":
                try:
                    label_text_cxsmi = (
                        "<cxsmi>"
                        + re.search(
                            re.escape("<cxsmi>") + r"(.*?)" + re.escape("</cxsmi>"),
                            label_text,
                        ).group(1)
                        + "</cxsmi>"
                    )
                    gt_smiles_opt = (
                        label_text_cxsmi.replace("<cxsmi>", "")
                        .replace("</cxsmi>", "")
                        .replace("</s>", "")
                        .replace(" ", "")
                    )
                except:
                    gt_smiles_opt = None
                gt_stable = markush_tokenizer.get_stable(label_text)
                print(gt_smiles_opt)
                gt_smiles = cxsmiles_tokenizer.convert_opt_to_out(gt_smiles_opt)
                if verbose:
                    print(f"gt_smiles: {gt_smiles}")
                if verbose:
                    print(f"gt_smiles_opt: {gt_smiles_opt}")
                try:
                    gt_smiles = cxsmiles_tokenizer.convert_opt_to_out(gt_smiles_opt)
                    if verbose:
                        print(f"gt_smiles: {gt_smiles}")
                except Exception as e:
                    if verbose:
                        print(
                            f"Error {e} in cxsmiles_tokenizer.convert_opt_to_out for {gt_smiles_opt}"
                        )
                    gt_smiles = None

            if (gt_smiles != None) and (Chem.MolFromSmiles(gt_smiles) == None):
                print(f"GT smiles can not be read: {gt_smiles}")
                # If the ground-truth SMILES is invalid, replace it with None
                predicted_smiles_list.append(None)
                gt_smiles_list.append(None)
                if (config["name"] == "ocxsr") or (config["name"] == "mdu"):
                    predicted_smiles_opt_list.append(None)
                    gt_smiles_opt_list.append(None)
                if config["name"] == "mdu":
                    predicted_stable_list.append(None)
                    gt_stable_list.append(None)
                continue

            gt_smiles_list.append(gt_smiles)
            if (config["name"] == "ocxsr") or (config["name"] == "mdu"):
                gt_smiles_opt_list.append(gt_smiles_opt)
            if config["name"] == "mdu":
                gt_stable_list.append(gt_stable)

            if hasattr(model, "module"):
                predicted_ids = model.module.generate(
                    **encoding, num_beams=1, max_length=512
                )  # Distributed training
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    transformers_logging.set_verbosity_error()
                    if model_args.beam_search:
                        predicted_ids = model.generate(
                            **encoding, num_beams=5, max_length=512
                        )  
                    else:
                        predicted_ids = model.generate(
                            **encoding, num_beams=1, max_length=512
                        )
                    transformers_logging.set_verbosity_info()

            # Quick test!
            if config["udop_tokenizer_only"]:
                predicted_text = tokenizer.decode(predicted_ids[0][1:-1])
                predicted_text = (
                    predicted_text.replace("<unk>", "<")
                    .replace("! [ [ 0, 0 ] ]", "")
                    .replace(" ", "")
                )
                print("Prediction:", predicted_text)
            else:
                predicted_text = (
                    markush_tokenizer_training.decode_plus_decode_other_tokens(
                        predicted_ids[0][1:-1]
                    )
                )
            if verbose:
                print(f"predicted_text: {predicted_text}")
            if config["name"] == "ocsr":
                predicted_smiles = (
                    predicted_text.replace("<smi>", "")
                    .replace("</smi>", "")
                    .replace("</s>", "")
                    .replace(" ", "")
                )
            if config["name"] == "ocxsr":
                predicted_smiles_opt = (
                    predicted_text.replace("<cxsmi>", "")
                    .replace("</cxsmi>", "")
                    .replace("</s>", "")
                    .replace(" ", "")
                )
                if verbose:
                    print(f"predicted_smiles_opt: {predicted_smiles_opt}")
                try:
                    predicted_smiles = cxsmiles_tokenizer_training.convert_opt_to_out(
                        predicted_smiles_opt
                    )
                    if verbose:
                        print(f"predicted_smiles: {predicted_smiles}")
                except Exception as e:
                    if verbose:
                        print(
                            f"Error {e} in convert_opt_to_out() for {predicted_smiles_opt}"
                        )
                    predicted_smiles = None
            if config["name"] == "mdu":
                try:
                    predicted_text_cxsmi = (
                        "<cxsmi>"
                        + re.search(
                            re.escape("<cxsmi>") + r"(.*?)" + re.escape("</cxsmi>"),
                            predicted_text,
                        ).group(1)
                        + "</cxsmi>"
                    )
                    predicted_smiles_opt = (
                        predicted_text_cxsmi.replace("<cxsmi>", "")
                        .replace("</cxsmi>", "")
                        .replace("</s>", "")
                        .replace(" ", "")
                    )
                except:
                    predicted_smiles_opt = None

                predicted_stable = markush_tokenizer_training.get_stable(predicted_text)
                if config["name"] == "mdu":
                    predicted_stable_list.append(predicted_stable)

                if verbose:
                    print(f"predicted_smiles_opt: {predicted_smiles_opt}")

                try:
                    predicted_smiles = cxsmiles_tokenizer_training.convert_opt_to_out(
                        predicted_smiles_opt
                    )
                    if verbose:
                        print(f"predicted_smiles: {predicted_smiles}")
                except Exception as e:
                    if verbose:
                        print(
                            f"Error {e} in convert_opt_to_out() for {predicted_smiles_opt}"
                        )
                    predicted_smiles = None

            # Display debug examples (incorrect gt are not displayed)
            if display_eval_samples and (idx < max_display_eval_samples):
                display_eval_sample(
                    image,
                    encoding["bbox"],
                    encoding["input_ids"][0],
                    input_text,
                    label_text,
                    predicted_text,
                    gt_smiles,
                    gt_smiles_opt,
                    predicted_smiles,
                    predicted_smiles_opt,
                    gt_stable,
                    predicted_stable,
                    config,
                    output_path=f"{display_samples_output_dir}/{idx}.png",
                    tokenizer=tokenizer,
                    display_errors=display_errors,
                    display_markush_evaluation=display_markush_evaluation,
                )

            if (predicted_smiles != None) and (
                Chem.MolFromSmiles(predicted_smiles) == None
            ):
                # If the predicted SMILES is invalid, replace it with None
                predicted_smiles_list.append(None)
                if (config["name"] == "ocxsr") or (config["name"] == "mdu"):
                    predicted_smiles_opt_list.append(None)
                continue

            predicted_smiles_list.append(predicted_smiles)
            if (config["name"] == "ocxsr") or (config["name"] == "mdu"):
                predicted_smiles_opt_list.append(predicted_smiles_opt)

    # Postprocess predicted SMILES
    molecule_postprocessor = MoleculePostprocessor()
    predicted_smiles_list = [
        molecule_postprocessor.postprocess(predicted_smiles)
        for predicted_smiles in predicted_smiles_list
    ]

    if config["name"] == "ocsr":
        scores_ar = get_scores(
            gt_smiles_list,
            predicted_smiles_list,
            training_smiles,
            get_unreduced_scores=False,
        )
        print(
            "Number of samples for autoregressive evaluation:",
            len([s for s in gt_smiles_list if s != None]),
        )
        metrics[metrics_prefix + "ar_valid"] = scores_ar["valid"]
        metrics[metrics_prefix + "ar_tanimoto"] = scores_ar["tanimoto"]
        metrics[metrics_prefix + "ar_is_in_training"] = scores_ar["is_in_training"]
        metrics[metrics_prefix + "ar_inchi_equality"] = scores_ar["inchi_equality"]
        metrics[metrics_prefix + "ar_string_equality"] = scores_ar["string_equality"]

    if config["name"] == "ocxsr":
        scores_ar = get_scores(
            gt_smiles_list,
            predicted_smiles_list,
            training_smiles,
            cxsmiles=True,
            get_unreduced_scores=False,
        )
        print(
            "Number of samples for autoregressive evaluation:",
            len([s for s in gt_smiles_list if s != None]),
        )
        metrics[metrics_prefix + "ar_valid"] = scores_ar["valid"]
        metrics[metrics_prefix + "ar_tanimoto"] = scores_ar["tanimoto"]
        metrics[metrics_prefix + "ar_is_in_training"] = scores_ar["is_in_training"]
        metrics[metrics_prefix + "ar_r"] = scores_ar["r"]
        metrics[metrics_prefix + "ar_m"] = scores_ar["m"]
        metrics[metrics_prefix + "ar_sg"] = scores_ar["sg"]
        metrics[metrics_prefix + "ar_r_size"] = scores_ar["r_size"]
        metrics[metrics_prefix + "ar_m_size"] = scores_ar["m_size"]
        metrics[metrics_prefix + "ar_sg_size"] = scores_ar["sg_size"]
        metrics[metrics_prefix + "ar_inchi_equality"] = scores_ar["inchi_equality"]
        metrics[metrics_prefix + "ar_cxsmi_equality"] = scores_ar["cxsmi_equality"]
        metrics[metrics_prefix + "ar_string_equality"] = scores_ar["string_equality"]

        number_correct_predictions = sum(
            (
                pred == gt
                for pred, gt in zip(predicted_smiles_opt_list, gt_smiles_opt_list)
                if gt != None
            )
        )
        number_ground_truths = len([gt for gt in gt_smiles_opt_list if gt != None])
        if number_ground_truths == 0:
            metrics[metrics_prefix + "cxsmi_ar_string_equality_opt"] = 0
        else:
            metrics[metrics_prefix + "cxsmi_ar_string_equality_opt"] = np.round(
                number_correct_predictions / number_ground_truths, 3
            )

    if config["name"] == "mdu":
        # Debug
        print(f"Finishing mdu evaluation with prefix: {metrics_prefix}")
        print(f"Len gt_smiles: {len(gt_smiles_list)}")
        print(f"Len predicted_smiles_list: {len(predicted_smiles_list)}")
        print(f"Len gt_stable_list: {len(gt_stable_list)}")
        print(f"Len predicted_stable_list: {len(predicted_stable_list)}")

        scores_ar = get_scores(
            gt_smiles_list,
            predicted_smiles_list,
            training_smiles,
            gt_stable_list,
            predicted_stable_list,
            cxsmiles=True,
            markush=True,
            get_unreduced_scores=False,
        )
        metrics[metrics_prefix + "ar_valid"] = scores_ar["valid"]
        metrics[metrics_prefix + "ar_valid"] = scores_ar["valid"]
        metrics[metrics_prefix + "ar_tanimoto"] = scores_ar["tanimoto"]
        metrics[metrics_prefix + "ar_is_in_training"] = scores_ar["is_in_training"]
        metrics[metrics_prefix + "ar_r"] = scores_ar["r"]
        metrics[metrics_prefix + "ar_m"] = scores_ar["m"]
        metrics[metrics_prefix + "ar_sg"] = scores_ar["sg"]
        metrics[metrics_prefix + "ar_r_size"] = scores_ar["r_size"]
        metrics[metrics_prefix + "ar_m_size"] = scores_ar["m_size"]
        metrics[metrics_prefix + "ar_sg_size"] = scores_ar["sg_size"]
        metrics[metrics_prefix + "ar_inchi_equality"] = scores_ar["inchi_equality"]
        metrics[metrics_prefix + "ar_cxsmi_equality"] = scores_ar["cxsmi_equality"]
        metrics[metrics_prefix + "ar_string_equality"] = scores_ar["string_equality"]
        metrics[metrics_prefix + "ar_stable_recall"] = scores_ar["stable_recall"]
        metrics[metrics_prefix + "ar_stable_precision"] = scores_ar["stable_precision"]
        metrics[metrics_prefix + "ar_stable_equality"] = scores_ar["stable_equality"]
        metrics[metrics_prefix + "ar_markush_equality"] = scores_ar["markush_equality"]
        metrics[metrics_prefix + "ar_size"] = scores_ar["size"]
        metrics[metrics_prefix + "invalid_gt"] = scores_ar["invalid_gt"]

        number_correct_predictions = sum(
            (
                pred == gt
                for pred, gt in zip(predicted_smiles_opt_list, gt_smiles_opt_list)
                if gt != None
            )
        )
        number_ground_truths = len([gt for gt in gt_smiles_opt_list if gt != None])
        if number_ground_truths == 0:
            metrics[metrics_prefix + "ar_string_equality_opt"] = 0
        else:
            metrics[metrics_prefix + "ar_string_equality_opt"] = np.round(
                number_correct_predictions / number_ground_truths, 3
            )

    if overwrite_predictions:
        with open(
            f"{training_args.output_dir}/{metrics_prefix}gt_smiles_list_{max_eval_samples}.pkl",
            "wb",
        ) as f:
            pickle.dump(gt_smiles_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(
            f"{training_args.output_dir}/{metrics_prefix}gt_smiles_opt_list_{max_eval_samples}.pkl",
            "wb",
        ) as f:
            pickle.dump(gt_smiles_opt_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(
            f"{training_args.output_dir}/{metrics_prefix}predicted_smiles_list_{max_eval_samples}.pkl",
            "wb",
        ) as f:
            pickle.dump(predicted_smiles_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(
            f"{training_args.output_dir}/{metrics_prefix}predicted_smiles_opt_list_{max_eval_samples}.pkl",
            "wb",
        ) as f:
            pickle.dump(predicted_smiles_opt_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save scores
    if save_scores:
        with open(
            f"{training_args.output_dir}/{metrics_prefix}scores_{max_eval_samples}.json",
            "w",
        ) as f:
            json.dump(metrics, f)
    return metrics


def get_stable_score(
    gt_stable, predicted_stable, permissive=True, verbose=False, normalize=True
):
    if verbose:
        print("gt_stable", gt_stable)
        print("predicted_stable", predicted_stable)
    scores = {"stable_equality": False, "stable_recall": 0.0, "stable_precision": 0.0}
    if predicted_stable is None:
        return scores
    if gt_stable == {}:
        print("GT substituent table is empty (the image is probably a molecule)")
        if predicted_stable == {}:
            scores["stable_equality"] = True
            scores["stable_recall"] = 1.0
            scores["stable_precision"] = 1.0
            return scores
        else:
            return scores

    if normalize:
        # Correct the prediction if only filler words are missing.
        # "aryl" to "an aryl group".
        # "nitrogen" to "a nitrogen".
        new_predicted_stable = {}
        for label, predicted_substituents in predicted_stable.items():
            if not (label in gt_stable):
                new_predicted_stable[label] = predicted_substituents
                continue
            new_predicted_substituents = []
            normalized_gt_substituents = [
                s.replace("a ", "").replace(" group", "") for s in gt_stable[label]
            ]
            for predicted_substituent in predicted_substituents:
                if predicted_substituent in gt_stable[label]:
                    new_predicted_substituents.append(predicted_substituent)
                    continue
                normalized_predicted_substituent = predicted_substituent.replace(
                    "a ", ""
                ).replace(" group", "")
                if not (normalized_predicted_substituent in normalized_gt_substituents):
                    new_predicted_substituents.append(predicted_substituent)
                    continue
                new_predicted_substituents.append(
                    gt_stable[label][
                        normalized_gt_substituents.index(
                            normalized_predicted_substituent
                        )
                    ]
                )
            new_predicted_stable[label] = new_predicted_substituents
        predicted_stable = new_predicted_stable
    if permissive:
        gt_stable = {
            k.lower(): [e.lower().replace(" ", "") for e in v]
            for k, v in gt_stable.items()
        }
        predicted_stable = {
            k.lower(): [e.lower().replace(" ", "") for e in v]
            for k, v in predicted_stable.items()
        }

    if verbose:
        print("gt_stable", gt_stable)
        print("predicted_stable", predicted_stable)
    # Compute Recall
    gt_found = []
    perfect_match = []
    for label, gt_substituents in gt_stable.items():
        if not (label in predicted_stable):
            perfect_match.append(False)
            gt_found.append([False] * len(gt_substituents))
            continue

        if set(gt_substituents) == set(predicted_stable[label]):
            perfect_match.append(True)
        else:
            perfect_match.append(False)

        gt_found_row = []
        for gt_substituent in gt_substituents:
            if gt_substituent in predicted_stable[label]:
                gt_found_row.append(True)
            else:
                gt_found_row.append(False)
        gt_found.append(gt_found_row)

    predicted_found = []
    for label, predicted_substituents in predicted_stable.items():
        if predicted_substituents == []:
            continue
        if not (label in gt_stable):
            predicted_found.append([False] * len(predicted_substituents))
            continue

        predicted_found_row = []
        for predicted_substituent in predicted_substituents:
            if predicted_substituent in gt_stable[label]:
                predicted_found_row.append(True)
            else:
                predicted_found_row.append(False)
        predicted_found.append(predicted_found_row)

    # Aggregate scores for each sample
    if all([s == True for s in perfect_match]):
        scores["stable_equality"] = True
    if verbose:
        print(
            "Stable Recall:",
            np.mean(
                [sum(gt_found_row) / len(gt_found_row) for gt_found_row in gt_found]
            ),
        )
        print(
            "Stable Precision:",
            np.mean(
                [
                    sum(predicted_found_row) / len(predicted_found_row)
                    for predicted_found_row in predicted_found
                ]
            ),
        )
    scores["stable_recall"] = round(
        np.mean([sum(gt_found_row) / len(gt_found_row) for gt_found_row in gt_found]), 3
    )
    scores["stable_precision"] = round(
        np.mean(
            [
                sum(predicted_found_row) / len(predicted_found_row)
                for predicted_found_row in predicted_found
            ]
        ),
        3,
    )

    if isinstance(scores["stable_precision"], float) and math.isnan(
        scores["stable_precision"]
    ):
        # predicted_stable = {}
        scores["stable_precision"] = 0.0

    return scores


def get_molecule_information(cxsmiles):
    information = {
        "r": False,
        "m": False,
        "sg": False,
    }
    parser_params = Chem.SmilesParserParams()
    parser_params.strictCXSMILES = False
    parser_params.sanitize = False
    parser_params.removeHs = False
    molecule = Chem.MolFromSmiles(cxsmiles, parser_params)
    # R
    rgroups = {}
    for i, atom in enumerate(molecule.GetAtoms()):
        if atom.HasProp("atomLabel"):
            rgroups[i] = atom.GetProp("atomLabel")
    if len(rgroups) > 0:
        information["r"] = True
    # M
    cxsmiles_tokenizer = CXSMILESTokenizer()
    m_sections = []
    if len(cxsmiles.split("|")) > 1:
        for section in cxsmiles_tokenizer.parse_sections(cxsmiles.split("|")[1]):
            if (len(section) >= 1) and not (section[0] == "m"):
                continue
            m_section = cxsmiles_tokenizer.parse_m_section(section)
            m_sections.append(
                {
                    "ring_atoms": [int(idx) for idx in m_section[2:] if idx != "."],
                    "atom_connector": int(m_section[1]),
                }
            )
    if len(m_sections) > 0:
        information["m"] = True
    # Sg
    gt_sgroups = Chem.rdchem.GetMolSubstanceGroups(molecule)
    if len(gt_sgroups) > 0:
        information["sg"] = True
    return information


def get_scores(
    gt_smiles_list,
    predicted_smiles_list,
    training_smiles,
    gt_stable_list=None,
    predicted_stable_list=None,
    cxsmiles=False,
    markush=False,
    get_unreduced_scores=True,
    verbose=False,
):
    """Get scores: validity, correctness, tanimoto"""
    scores = {}
    for id, (gt_smiles, predicted_smiles) in tqdm(
        enumerate(zip(gt_smiles_list, predicted_smiles_list)), total=len(gt_smiles_list)
    ):
        if gt_smiles is None:
            print(f"GT molecule is None: {id, gt_smiles}")
            scores[id] = None
            continue
        parser_params = Chem.SmilesParserParams()
        parser_params.strictCXSMILES = False
        parser_params.sanitize = False
        parser_params.removeHs = False
        gt_molecule = Chem.MolFromSmiles(gt_smiles, parser_params)
        if gt_molecule is None:
            print(f"GT molecule is None: {id, gt_smiles}")
            scores[id] = None
            continue

        default_incorrect_score = {
            "levenshtein": len(gt_smiles),
            "levenshtein0": False,
            "tanimoto": 0.0,
            "tanimoto1": False,
            "bleu_average": 0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rouge3": 0.0,
            "rouge4": 0.0,
            "rougeL": 0.0,
            "valid": False,
            "inchi_equality": False,
            "string_equality": False,
            "r": 0.0,
            "m": 0.0,
            "sg": 0.0,
            "cxsmi_equality": False,
            "markush_equality": False,
        }
        molecule_information = get_molecule_information(gt_smiles)
        if molecule_information["r"] == False:
            default_incorrect_score["r"] = None
        if molecule_information["m"] == False:
            default_incorrect_score["m"] = None
        if molecule_information["sg"] == False:
            default_incorrect_score["sg"] = None

        if predicted_smiles is None:
            scores[id] = default_incorrect_score
            continue
        if not (cxsmiles):
            gt_smiles = Chem.MolToSmiles(gt_molecule)
        else:
            gt_smiles = canonicalize_markush(gt_smiles)
            if verbose:
                print(f"gt_smiles: {gt_smiles}")
        if gt_smiles is None:
            print(f"GT molecule is None (after canonicalization): {id, gt_smiles}")
            scores[id] = None
            continue

        predicted_molecule = Chem.MolFromSmiles(predicted_smiles, parser_params)
        if predicted_molecule is None:
            scores[id] = default_incorrect_score
            continue
        if not (cxsmiles) and not (markush):
            predicted_smiles = Chem.MolToSmiles(predicted_molecule)
        else:
            try:
                predicted_smiles = canonicalize_markush(predicted_smiles)
            except:
                scores[id] = default_incorrect_score
                continue
        if predicted_smiles is None:
            scores[id] = default_incorrect_score
            continue

        if cxsmiles or markush:
            try:
                scores[id] = compute_markush_prediction_quality(
                    predicted_smiles,
                    gt_smiles,
                    remove_stereo=True,
                    remove_double_bond_stereo=True,
                )
            except Exception as e:
                print(
                    f"Error {e} in compute_markush_prediction_quality (in get_scores()) for: gt: {gt_smiles} and prediction: {predicted_smiles}"
                )
                scores[id] = default_incorrect_score

    if markush:
        for id, (gt_smiles, predicted_smiles) in tqdm(
            enumerate(zip(gt_smiles_list, predicted_smiles_list)),
            total=len(gt_smiles_list),
        ):
            if scores[id] == None:
                continue

            if gt_stable_list[id] is None:
                print(f"For id: {id}, gt substituent table is None.")
                if id not in scores:
                    scores[id] = None
                scores[id]["stable_equality"] = None
                scores[id]["stable_recall"] = None
                scores[id]["stable_precision"] = None
                scores[id]["markush_equality"] = None
                continue

            stable_scores = get_stable_score(
                gt_stable_list[id], predicted_stable_list[id]
            )
            scores[id]["stable_equality"] = stable_scores["stable_equality"]
            scores[id]["stable_recall"] = stable_scores["stable_recall"]
            scores[id]["stable_precision"] = stable_scores["stable_precision"]
            scores[id]["markush_equality"] = (
                scores[id]["cxsmi_equality"] and scores[id]["stable_equality"]
            )

    # Evaluate training set overfitting
    for id, predicted_smiles in enumerate(predicted_smiles_list):
        if scores[id] is None:
            continue
        if predicted_smiles is None:
            scores[id]["is_in_training"] = 0
            continue
        scores[id]["is_in_training"] = int(predicted_smiles in training_smiles)

    # Reduce scores
    reduced_scores = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reduced_scores["tanimoto"] = np.round(
            np.mean(
                [scores[id]["tanimoto"] for id in scores.keys() if scores[id] != None]
            ),
            3,
        )
        reduced_scores["valid"] = np.round(
            np.mean(
                [scores[id]["valid"] for id in scores.keys() if scores[id] != None]
            ),
            3,
        )
        reduced_scores["inchi_equality"] = np.round(
            np.mean(
                [
                    scores[id]["inchi_equality"]
                    for id in scores.keys()
                    if scores[id] != None
                ]
            ),
            3,
        )
        reduced_scores["is_in_training"] = np.round(
            np.mean(
                [
                    scores[id]["is_in_training"]
                    for id in scores.keys()
                    if scores[id] != None
                ]
            ),
            3,
        )
        reduced_scores["string_equality"] = np.round(
            np.mean(
                [
                    scores[id]["string_equality"]
                    for id in scores.keys()
                    if scores[id] != None
                ]
            ),
            3,
        )
    reduced_scores["invalid_gt"] = sum(
        (scores[id] == None) for id in scores.keys()
    ) / len(scores)
    reduced_scores["size"] = len(scores)

    if cxsmiles:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reduced_scores["r"] = np.round(
                np.mean(
                    [
                        scores[id]["r"]
                        for id in scores.keys()
                        if (scores[id] != None) and (scores[id]["r"] != None)
                    ]
                ),
                3,
            )
            reduced_scores["m"] = np.round(
                np.mean(
                    [
                        scores[id]["m"]
                        for id in scores.keys()
                        if (scores[id] != None) and (scores[id]["m"] != None)
                    ]
                ),
                3,
            )
            reduced_scores["sg"] = np.round(
                np.mean(
                    [
                        scores[id]["sg"]
                        for id in scores.keys()
                        if (scores[id] != None) and (scores[id]["sg"] != None)
                    ]
                ),
                3,
            )
            reduced_scores["r_size"] = len(
                [
                    True
                    for id in scores.keys()
                    if (scores[id] != None) and (scores[id]["r"] != None)
                ]
            )
            if verbose:
                for i, k in scores.items():
                    print("score", i, scores[i])
                    if scores[i]["m"] == None:
                        print("score (m none)", i, scores[i])
                    if scores[id] == None:
                        print("score (none)", i, scores[i])
            reduced_scores["m_size"] = len(
                [
                    True
                    for id in scores.keys()
                    if (scores[id] != None) and (scores[id]["m"] != None)
                ]
            )
            reduced_scores["sg_size"] = len(
                [
                    True
                    for id in scores.keys()
                    if (scores[id] != None) and (scores[id]["sg"] != None)
                ]
            )
            reduced_scores["cxsmi_equality"] = np.round(
                np.mean(
                    [
                        scores[id]["cxsmi_equality"]
                        for id in scores.keys()
                        if scores[id] != None
                    ]
                ),
                3,
            )

    if markush:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for id in range(len(scores)):
                if scores[id] == None:
                    continue
            reduced_scores["stable_equality"] = np.round(
                np.mean(
                    [
                        scores[id]["stable_equality"]
                        for id in scores.keys()
                        if (scores[id] != None)
                        and (scores[id]["stable_equality"] != None)
                    ]
                ),
                3,
            )
            reduced_scores["stable_recall"] = np.round(
                np.mean(
                    [
                        scores[id]["stable_recall"]
                        for id in scores.keys()
                        if (scores[id] != None)
                        and (scores[id]["stable_recall"] != None)
                    ]
                ),
                3,
            )
            reduced_scores["stable_precision"] = np.round(
                np.mean(
                    [
                        scores[id]["stable_precision"]
                        for id in scores.keys()
                        if (scores[id] != None)
                        and (scores[id]["stable_precision"] != None)
                    ]
                ),
                3,
            )
            reduced_scores["markush_equality"] = np.round(
                np.mean(
                    [
                        scores[id]["markush_equality"]
                        for id in scores.keys()
                        if (scores[id] != None)
                        and (scores[id]["markush_equality"] != None)
                    ]
                ),
                3,
            )

    if get_unreduced_scores:
        return reduced_scores, scores
    else:
        return reduced_scores


def compute_molecule_prediction_quality(
    predicted_smiles,
    gt_smiles,
    predicted_molecule=None,
    gt_molecule=None,
    remove_stereo=False,
    remove_double_bond_stereo=True,
    compute_nlp_metrics=False,
):
    scores = {
        "levenshtein": len(gt_smiles),
        "levenshtein0": False,
        "tanimoto": 0,
        "tanimoto1": False,
        "bleu_average": 0,
        "bleu1": 0,
        "bleu2": 0,
        "bleu3": 0,
        "bleu4": 0,
        "rouge1": 0,
        "rouge2": 0,
        "rouge3": 0,
        "rouge4": 0,
        "rougeL": 0,
        "valid": False,
        "inchi_equality": False,
        "string_equality": False,
    }
    # print("predicted_mol input", predicted_smiles)
    # print("gt_mol input", gt_smiles)

    if predicted_smiles is None or (
        isinstance(predicted_smiles, float) and math.isnan(predicted_smiles)
    ):
        return scores

    if Chem.MolFromSmiles(predicted_smiles) is None:
        return scores

    scores["string_equality"] = predicted_smiles == gt_smiles

    if compute_nlp_metrics:
        import Levenshtein
        
        # Levenshtein distance
        levenshtein = Levenshtein.distance(predicted_smiles, gt_smiles)
        scores["levenshtein"] = levenshtein
        if levenshtein == 0:
            scores["levenshtein0"] = True

    # Tanimoto score
    # Note: get_molecule_from_smiles(kekulize = True) can cause single/double bonds mismatches.
    # For instance: "Bc1ccc(-c2cc3cccnc3cc2O)nn1", remove_stereochemistry = False)" and "BC1=NN=C(C2=CC3=C(C=C2O)N=CC=C3)C=C1"
    # But, on the other hand, by aromatizing and kekulizing back cycles with R-groups, we can lose some double bonds.
    # For instance: gt_smiles = "[1*]C.[2*]C.[3*]C(C)(C)C(CC(=O)O)NC1=C(F)C=[4*]C(C2=CNC3=[5*]C=[6*]C=C32)=N1 |$R1;;R2;;R3;;;;;;;;;;;;;;X;;;;;;Y;;Z;;;$,m:1:23.24.25.26.27.28,m:3:14.15.17.18.19.29|"
    if not predicted_molecule:
        predicted_molecule = get_molecule_from_smiles(
            predicted_smiles, remove_stereochemistry=remove_stereo, kekulize=True
        )
    if not gt_molecule:
        gt_molecule = get_molecule_from_smiles(
            gt_smiles, remove_stereochemistry=remove_stereo, kekulize=True
        )  # Kekulize = True !

    if gt_molecule is None:
        print(
            f"Error: gt_molecule is 'None' in compute_molecule_prediction_quality for: gt: {gt_smiles} and prediction: {predicted_smiles}"
        )
        return scores

    # Remove hydrogens (They are only useful for stereo-chemistry)
    if remove_stereo:
        # print("predicted_mol", Chem.MolToSmiles(predicted_molecule))
        # print("gt_mol", Chem.MolToSmiles(gt_molecule))
        predicted_molecule = Chem.RemoveHs(predicted_molecule)
        gt_molecule = Chem.RemoveHs(gt_molecule)

    if remove_double_bond_stereo:
        for bond in gt_molecule.GetBonds():
            bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
        for bond in predicted_molecule.GetBonds():
            bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)

    # Remove aromatic bonds (Kekulize)
    gt_molecule = rdMolDraw2D.PrepareMolForDrawing(gt_molecule, addChiralHs=False)
    predicted_molecule = rdMolDraw2D.PrepareMolForDrawing(
        predicted_molecule, addChiralHs=False
    )

    scores["tanimoto"] = DataStructs.FingerprintSimilarity(
        Chem.RDKFingerprint(gt_molecule), Chem.RDKFingerprint(predicted_molecule)
    )
    scores["tanimoto1"] = scores["tanimoto"] == 1

    # Inchi equality
    if remove_stereo:
        gt_inchi, predicted_inchi = MolToInchi(
            predicted_molecule, options="/SNon"
        ), MolToInchi(gt_molecule, options="/SNon")
    else:
        gt_inchi, predicted_inchi = MolToInchi(gt_molecule), MolToInchi(
            predicted_molecule
        )

    if (gt_inchi != "") and (gt_inchi == predicted_inchi):
        scores["inchi_equality"] = True

    if compute_nlp_metrics:
        # BLEU score
        scores["bleu_average"] = sentence_bleu(
            [[c for c in gt_smiles]],
            [c for c in predicted_smiles],
            weights=[0.25, 0.25, 0.25, 0.25],
            smoothing_function=SmoothingFunction().method1,
        )
        scores["bleu1"] = sentence_bleu(
            [[c for c in gt_smiles]],
            [c for c in predicted_smiles],
            weights=[1, 0, 0, 0],
            smoothing_function=SmoothingFunction().method1,
        )
        scores["bleu2"] = sentence_bleu(
            [[c for c in gt_smiles]],
            [c for c in predicted_smiles],
            weights=[0, 1, 0, 0],
            smoothing_function=SmoothingFunction().method1,
        )
        scores["bleu3"] = sentence_bleu(
            [[c for c in gt_smiles]],
            [c for c in predicted_smiles],
            weights=[0, 0, 1, 0],
            smoothing_function=SmoothingFunction().method1,
        )
        scores["bleu4"] = sentence_bleu(
            [[c for c in gt_smiles]],
            [c for c in predicted_smiles],
            weights=[0, 0, 0, 1],
            smoothing_function=SmoothingFunction().method1,
        )

        # ROUGE score
        scores_rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rouge3", "rouge4", "rougeL"], use_stemmer=True
        ).score(
            " ".join([c for c in gt_smiles]), " ".join([c for c in predicted_smiles])
        )
        scores["rouge1"] = scores_rouge["rouge1"].fmeasure
        scores["rouge2"] = scores_rouge["rouge2"].fmeasure
        scores["rouge3"] = scores_rouge["rouge3"].fmeasure
        scores["rouge4"] = scores_rouge["rouge4"].fmeasure
        scores["rougeL"] = scores_rouge["rougeL"].fmeasure

    # Valid SMILES
    scores["valid"] = True

    return scores


def get_smiles_star_raw(
    smiles,
    keypoints,
    mol=None,
    encode_position=False,
    grounded_smiles=False,
    ocr_box_size=80,
):
    """
    keypoints = [[keypoints[i] - 1, keypoints[i+1] - 1] for i in range(0, len(keypoints), 3)]
    """
    if not (encode_position):
        return smiles + "![[0,0]]"
        # return smiles + "!" #+ ",".join([str(k) for k in keypoints])

    molecule = rdmolfiles.MolFromMolBlock(mol, removeHs=False)
    loc_cells = []
    for atom, keypoint in zip(molecule.GetAtoms(), keypoints):
        text = atom.GetSymbol()
        if atom.GetFormalCharge() != 0:
            if "-" in str(atom.GetFormalCharge()):
                text += "-"
            else:
                text += "+"
            text += str(atom.GetFormalCharge()).replace("-", "").replace("+", "")
        loc_cells.append(
            {
                "bbox": [
                    (keypoint[0] - ocr_box_size / 2),
                    (keypoint[1] - ocr_box_size / 2),
                    (keypoint[0] + ocr_box_size / 2),
                    (keypoint[1] + ocr_box_size / 2),
                ],
                "text": text,
            }
        )
    atom_boxes_str = ",".join([str(loc_cell["bbox"]) for loc_cell in loc_cells])
    if grounded_smiles:
        # Start smiles top left
        top_left_atom_index = 0
        min_dist = float("inf")
        for i, keypoint in enumerate(keypoints):
            d = math.dist([0, 0], keypoint)
            if d < min_dist:
                min_dist = d
                top_left_atom_index = i
        smiles = Chem.MolToSmiles(
            molecule, canonical=True, rootedAtAtom=top_left_atom_index
        )
        # Possible TODO: Modify atom boxes list to align with new smiles ordering
    smiles_star_raw = smiles + "!" + atom_boxes_str
    return smiles_star_raw


def replace_wildcards(cxsmiles, remove_stereo):
    parser_params = Chem.SmilesParserParams()
    parser_params.sanitize = False
    parser_params.strictCXSMILES = False
    parser_params.removeHs = False
    m = Chem.MolFromSmiles(cxsmiles, parser_params)
    Chem.SanitizeMol(
        m,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
    )  # lum # Kekulize
    if m is None:  # lum
        m = Chem.MolFromSmiles(cxsmiles, parser_params)  # lum

    for atom in m.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(6)
            atom.SetIsotope(0)
    smiles = Chem.MolToSmiles(m)
    m2 = get_molecule_from_smiles(smiles, remove_stereo)  # lum
    if m2 is None:  # lum
        m = Chem.MolFromSmiles(cxsmiles, parser_params)  # lum
        smiles = Chem.MolToSmiles(m)  # lum
    return smiles


def compute_markush_prediction_quality(
    predicted_smiles,
    gt_smiles,
    remove_stereo=False,
    remove_double_bond_stereo=True,
    verbose=False,
):
    """
    Note: Capitalization of R-groups labels is ignored (as example: "RB" = "Rb").
    Note: Markush structures with connections to cycles and fully-symetrical structures are problematic. They can be evaluated as correct, while they are not.
    For instance, "46.png" ("US9372402B2_19_2.png")
    """
    scores = {
        "backbone_core_tanimoto": 0,
        "backbone_core_tanimoto1": False,
        "backbone_core_inchi_equality": False,
        "backbone_fragments_tanimoto": [],
        "backbone_fragments_tanimoto1": [],
        "backbone_fragments_inchi_equality": [],
        "backbone_fragments_tanimoto_reduced": 0,
        "backbone_fragments_tanimoto1_reduced": False,
        "backbone_fragments_inchi_equality_reduced": False,
        "tanimoto": 0,
        "tanimoto1": False,
        "inchi_equality": False,
        "string_equality": False,
        "valid": False,
        "r_labels": [],
        "m_sections": [],
        "sg_sections": [],
        "r": 0,
        "m": 0,
        "sg": 0,
        "cxsmi_equality": False,
    }
    scores["string_equality"] = predicted_smiles == gt_smiles

    # Read molecules
    parser_params = Chem.SmilesParserParams()
    parser_params.strictCXSMILES = False
    parser_params.sanitize = False
    parser_params.removeHs = False
    predicted_molecule = Chem.MolFromSmiles(predicted_smiles, parser_params)
    gt_molecule = Chem.MolFromSmiles(gt_smiles, parser_params)

    # Get R groups
    gt_rgroups = {}
    for i, atom in enumerate(gt_molecule.GetAtoms()):
        if atom.HasProp("atomLabel"):
            gt_rgroups[i] = atom.GetProp("atomLabel")

    # Aromatize SMILES to avoid mismatches of kekulization on aromatic cycles
    # It requires to convert R-groups to carbons
    if verbose:
        print("gt_smiles before aromatization", Chem.MolToCXSmiles(gt_molecule))
        print(
            "predicted_smiles before aromatization",
            Chem.MolToCXSmiles(predicted_molecule),
        )
    for i, atom in enumerate(predicted_molecule.GetAtoms()):
        if atom.HasProp("atomLabel"):
            atom.SetAtomicNum(6)
    for i, atom in enumerate(predicted_molecule.GetAtoms()):
        if atom.HasProp("atomLabel"):
            atom.SetAtomicNum(6)
    Chem.SanitizeMol(
        predicted_molecule,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
    )
    Chem.SanitizeMol(
        gt_molecule,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
    )

    for i, atom in enumerate(gt_molecule.GetAtoms()):
        if atom.HasProp("atomLabel"):
            atom.SetAtomicNum(0)
    for i, atom in enumerate(predicted_molecule.GetAtoms()):
        if atom.HasProp("atomLabel"):
            atom.SetAtomicNum(0)
    if verbose:
        print("gt_smiles after aromatization", Chem.MolToCXSmiles(gt_molecule))
        print(
            "predicted_smiles after aromatization",
            Chem.MolToCXSmiles(predicted_molecule),
        )
    # Remove aromatic bonds.
    # Chem.Kekulize(predicted_molecule, clearAromaticFlags=True)
    # Chem.Kekulize(gt_molecule, clearAromaticFlags=True)

    # Get backbone scores
    predicted_fragments_indices = Chem.GetMolFrags(predicted_molecule)
    gt_fragments_indices = Chem.GetMolFrags(gt_molecule)
    predicted_fragments = []
    parser_params = Chem.SmilesParserParams()
    parser_params.sanitize = False
    parser_params.removeHs = False
    for predicted_fragment_indices in predicted_fragments_indices:
        predicted_fragment = Chem.MolFromSmiles(
            rdmolfiles.MolFragmentToSmiles(
                predicted_molecule, atomsToUse=predicted_fragment_indices
            ),
            parser_params,
        )
        Chem.SanitizeMol(
            predicted_fragment,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
        )  # lum
        predicted_fragments.append(predicted_fragment)

    gt_fragments = []
    for gt_fragment_indices in gt_fragments_indices:
        gt_fragment = Chem.MolFromSmiles(
            rdmolfiles.MolFragmentToSmiles(gt_molecule, atomsToUse=gt_fragment_indices),
            parser_params,
        )
        Chem.SanitizeMol(
            gt_fragment,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
        )  # lum
        gt_fragments.append(gt_fragment)
    core_gt_fragment_index = max(
        enumerate(gt_fragments), key=lambda m: m[1].GetNumAtoms()
    )[0]

    core_backbone_size = gt_fragments[core_gt_fragment_index].GetNumAtoms()
    fragments_backbone_total_size = sum(
        fragment.GetNumAtoms()
        for fragment in gt_fragments
        if (fragment.GetNumAtoms() != core_backbone_size)
    )

    fragments_mapping = defaultdict(list)
    fragments_indices_mapping = defaultdict(list)
    predicted_fragments_current = copy.deepcopy(predicted_fragments)
    predicted_fragments_indices_current = copy.deepcopy(predicted_fragments_indices)
    # Note: This approach is not optimal if the gt and prediction have different numbers of fragments.
    # For instance, let's say the gt has 2 fragments, the first one being "CC", and the prediction has only one. The predicted fragment would be matched to "CC".
    # However, it could have been a correct prediction of the second gt fragment.
    for i_gt, gt_fragment in enumerate(gt_fragments):
        if len(predicted_fragments_current) == 0:
            if verbose:
                print("len(predicted_fragments_current) == 0")
            # If all predicted fragments are already associated, create a fake one.
            predicted_fragment_smiles = ""
            selected_indices = []
        else:
            nb_atoms_found = []
            selected_index = 0
            # For each gt fragment, find the most similar predicted fragment
            for predicted_fragment in predicted_fragments_current:
                mcs = rdFMCS.FindMCS([predicted_fragment, gt_fragment], timeout=5)
                nb_atoms_found.append(mcs.numAtoms)
                if verbose:
                    print("MCS nb_atoms_found", nb_atoms_found)

            selected_indices = [
                i for i, j in enumerate(nb_atoms_found) if j == max(nb_atoms_found)
            ]
            selected_indices_new = selected_indices

            # Filter matches based on R-labels ([1*]C and [2*]C fragments can result in ties)
            if len(selected_indices) > 1:
                remove_indices = []
                for rgroup_idx, rgroup_label in gt_rgroups.items():
                    if not (rgroup_idx in gt_fragments_indices[i_gt]):
                        continue
                    for selected_index in selected_indices:
                        matched_rlabel = False
                        for i, atom in enumerate(predicted_molecule.GetAtoms()):
                            if not (
                                i in predicted_fragments_indices_current[selected_index]
                            ):
                                continue
                            if atom.HasProp("atomLabel"):
                                if (
                                    atom.GetProp("atomLabel").lower()
                                    == rgroup_label.lower()
                                ):
                                    matched_rlabel = True
                        if not (matched_rlabel):
                            remove_indices.append(selected_index)
                if verbose:
                    print("remove indices:", remove_indices)
                selected_indices_new = [
                    selected_index
                    for selected_index in selected_indices
                    if not (selected_index in remove_indices)
                ]

            # Fallback option, select the smallest fragment
            if len(selected_indices_new) == 0:
                min_length = float("inf")
                min_selected_index = None
                for selected_index in selected_indices:
                    if (
                        len(predicted_fragments_indices_current[selected_index])
                        < min_length
                    ):
                        min_length = len(
                            predicted_fragments_indices_current[selected_index]
                        )
                        min_selected_index = selected_index
                selected_indices_new = [min_selected_index]

            selected_indices = selected_indices_new

            # If multiple fragments are equally similar, compute metrics on only one arbitrarily.
            # Even if fragments are "shuffled", they are similar, so there is no impact on the computed backbone scores
            selected_index = selected_indices[0]
            predicted_fragment_smiles = Chem.MolToSmiles(
                predicted_fragments_current[selected_index]
            )

        gt_fragment_smiles = Chem.MolToSmiles(gt_fragment)

        # Replace wildcards
        if verbose:
            print("gt_fragment_smiles before replace_wildcards", gt_fragment_smiles)
            print(
                "predicted_fragment_smiles before replace_wildcards",
                predicted_fragment_smiles,
            )
        predicted_fragment_smiles = replace_wildcards(
            predicted_fragment_smiles, remove_stereo
        )
        gt_fragment_smiles = replace_wildcards(gt_fragment_smiles, remove_stereo)
        if verbose:
            print("gt_fragment_smiles after replace_wildcards", gt_fragment_smiles)
            print(
                "predicted_fragment_smiles after replace_wildcards",
                predicted_fragment_smiles,
            )

        if Chem.MolFromSmiles(gt_fragment_smiles) == None:
            print(
                f"gt_fragment_molecule is 'None' in compute_markush_prediction_quality() for: predicted_smiles: {predicted_smiles}, gt_smiles: {gt_smiles}"
            )

        # Get fragments score
        fragment_score = compute_molecule_prediction_quality(
            predicted_smiles=predicted_fragment_smiles,
            gt_smiles=gt_fragment_smiles,
            remove_stereo=remove_stereo,
            remove_double_bond_stereo=remove_double_bond_stereo,
        )
        if verbose:
            print("fragment_score", fragment_score)

        if gt_fragment.GetNumAtoms() == core_backbone_size:
            scores["backbone_core_tanimoto"] = np.round(fragment_score["tanimoto"], 3)
            scores["backbone_core_tanimoto1"] = fragment_score["tanimoto1"]
            scores["backbone_core_inchi_equality"] = fragment_score["inchi_equality"]
        else:
            scores["backbone_fragments_tanimoto"].append(
                np.round(fragment_score["tanimoto"], 3)
            )
            scores["backbone_fragments_tanimoto1"].append(fragment_score["tanimoto1"])
            scores["backbone_fragments_inchi_equality"].append(
                fragment_score["inchi_equality"]
            )

        # Store mapping
        for selected_index in selected_indices:
            fragments_mapping[i_gt].append(predicted_fragments_current[selected_index])
            fragments_indices_mapping[i_gt].append(
                predicted_fragments_indices_current[selected_index]
            )
        if verbose:
            print("selected_indices", selected_indices)
        if len(selected_indices) == 1:
            # Consume matched predicted fragment
            predicted_fragments_current = [
                f
                for i, f in enumerate(predicted_fragments_current)
                if (i != selected_index)
            ]
            predicted_fragments_indices_current = [
                f
                for i, f in enumerate(predicted_fragments_indices_current)
                if (i != selected_index)
            ]

    if verbose:
        print("fragments_indices_mapping:", fragments_indices_mapping)
    # Reduce backbone scores
    if scores["backbone_fragments_tanimoto"] == []:
        scores["backbone_fragments_tanimoto_reduced"] = 0.0
    else:
        scores["backbone_fragments_tanimoto_reduced"] = np.round(
            np.mean(scores["backbone_fragments_tanimoto"]), 3
        )
    scores["backbone_fragments_tanimoto1_reduced"] = all(
        s == True for s in scores["backbone_fragments_tanimoto1"]
    )
    scores["backbone_fragments_inchi_equality_reduced"] = all(
        s == True for s in scores["backbone_fragments_inchi_equality"]
    )

    scores["tanimoto"] = np.round(
        (
            scores["backbone_fragments_tanimoto_reduced"]
            * fragments_backbone_total_size
            + scores["backbone_core_tanimoto"] * core_backbone_size
        )
        / (fragments_backbone_total_size + core_backbone_size),
        3,
    )
    scores["tanimoto1"] = all(
        s == True
        for s in [scores["backbone_fragments_tanimoto1_reduced"]]
        + [scores["backbone_core_tanimoto1"]]
    )
    scores["inchi_equality"] = all(
        s == True
        for s in [scores["backbone_fragments_inchi_equality_reduced"]]
        + [scores["backbone_core_inchi_equality"]]
    )

    # Create global mapping
    gt_to_pred_indices_mapping = defaultdict(list)
    for i_gt, gt_fragment in enumerate(gt_fragments):
        for predicted_fragment, predicted_fragments_indices in zip(
            fragments_mapping[i_gt], fragments_indices_mapping[i_gt]
        ):
            if verbose:
                print(
                    "predicted_fragment before FindMCS",
                    Chem.MolToCXSmiles(predicted_fragment),
                )
                print("gt_fragment before FindMCS", Chem.MolToCXSmiles(gt_fragment))

            mcs = rdFMCS.FindMCS([predicted_fragment, gt_fragment], timeout=5)
            mcs_molecule = Chem.MolFromSmarts(mcs.smartsString)
            if verbose:
                print("MCS smarts_string:", mcs.smartsString)
                print("MCS molecule", Chem.MolToCXSmiles(mcs_molecule))
                print(
                    "predicted_molecule before MCS search",
                    Chem.MolToCXSmiles(predicted_molecule),
                )
                print("gt_molecule before MCS search", Chem.MolToCXSmiles(gt_molecule))

            Chem.SanitizeMol(
                gt_molecule,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
            )  # lum
            Chem.SanitizeMol(
                predicted_molecule,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
            )  # lum
            gt_matches = gt_molecule.GetSubstructMatches(mcs_molecule, uniquify=False)
            predicted_matches = predicted_molecule.GetSubstructMatches(
                mcs_molecule, uniquify=False
            )

            if verbose:
                print("predicted_matches:")
                print(predicted_matches)
                print("gt_matches:")
                print(gt_matches)

            # Filter matches out of the fragment
            remove_indices = []
            for i, m in enumerate(predicted_matches):
                if not (all(i in predicted_fragments_indices for i in m)):
                    remove_indices.append(i)
            predicted_matches = [
                m for i, m in enumerate(predicted_matches) if not (i in remove_indices)
            ]
            if verbose:
                print("filtered predicted_matches:", predicted_matches)

            remove_indices = []
            for i, m in enumerate(gt_matches):
                if not (all(i_m in gt_fragments_indices[i_gt] for i_m in m)):  # lum
                    remove_indices.append(i)
            gt_matches = [
                m for i, m in enumerate(gt_matches) if not (i in remove_indices)
            ]
            if verbose:
                print("filtered gt_matches:", gt_matches)

            # Gt matches and predicted matches can be of different sizes
            for gt_match in gt_matches:
                for predicted_match in predicted_matches:
                    for m_i_pred, m_i_gt in zip(predicted_match, gt_match):
                        if not (m_i_pred in gt_to_pred_indices_mapping[m_i_gt]):
                            gt_to_pred_indices_mapping[m_i_gt].append(m_i_pred)
    if verbose:
        print("gt_to_pred_indices_mapping:")
        pprint(gt_to_pred_indices_mapping)

    if gt_rgroups == {}:
        scores["r_labels"] = None
    if verbose:
        print("gt_rgroups:", gt_rgroups)

    # Test R
    gt_to_pred_indices_mapping_r = copy.deepcopy(gt_to_pred_indices_mapping)
    for i, rgroup_label in gt_rgroups.items():
        correct = False
        if i in gt_to_pred_indices_mapping_r:
            for j in gt_to_pred_indices_mapping_r[i]:
                r_atom = predicted_molecule.GetAtomWithIdx(j)
                if r_atom.HasProp("atomLabel"):
                    if r_atom.GetProp("atomLabel").lower() == rgroup_label.lower():
                        correct = True
                        # Consume
                        gt_to_pred_indices_mapping_r = {
                            k: [idx for idx in v if (idx != j)]
                            for k, v in gt_to_pred_indices_mapping_r.items()
                        }
                        if verbose:
                            print(
                                f"prediction rgroup matched: {r_atom.GetProp('atomLabel').lower()}, {j} index in prediction, {i} index in gt"
                            )
                        break
        scores["r_labels"].append(correct)

    # Get m sections
    cxsmiles_tokenizer = CXSMILESTokenizer()
    m_sections_gt = []
    if len(gt_smiles.split("|")) > 1:
        for section in cxsmiles_tokenizer.parse_sections(gt_smiles.split("|")[1]):
            if (len(section) >= 1) and not (section[0] == "m"):
                continue
            m_section = cxsmiles_tokenizer.parse_m_section(section)
            m_sections_gt.append(
                {
                    "ring_atoms": [int(idx) for idx in m_section[2:] if idx != "."],
                    "atom_connector": int(m_section[1]),
                }
            )
    if verbose:
        print(f"m_sections_gt: {m_sections_gt}")

    if m_sections_gt == []:
        scores["m_sections"] = None

    # Get m sections predictions
    m_sections_predicted = []
    if len(predicted_smiles.split("|")) > 1:
        for section in cxsmiles_tokenizer.parse_sections(
            predicted_smiles.split("|")[1]
        ):
            if (len(section) >= 1) and not (section[0] == "m"):
                continue
            m_section = cxsmiles_tokenizer.parse_m_section(section)
            m_sections_predicted.append(
                {
                    "ring_atoms": [int(idx) for idx in m_section[2:] if idx != "."],
                    "atom_connector": int(m_section[1]),
                }
            )
    if verbose:
        print(f"m_sections_predicted: {m_sections_predicted}")

    # Test m
    gt_to_pred_indices_mapping_m = copy.deepcopy(gt_to_pred_indices_mapping)
    for m_section_gt in m_sections_gt:
        correct = False
        for m_section_predicted in m_sections_predicted:
            correct_atom_connector = False
            correct_atom_rings = False
            if (m_section_gt["atom_connector"] in gt_to_pred_indices_mapping_m) and (
                m_section_predicted["atom_connector"]
                in gt_to_pred_indices_mapping_m[m_section_gt["atom_connector"]]
            ):
                correct_atom_connector = True
            ring_atoms_found = []
            for ring_atom in m_section_gt["ring_atoms"]:
                found = False
                matched_indices = []
                if not (ring_atom in gt_to_pred_indices_mapping_m):
                    continue
                for i in gt_to_pred_indices_mapping_m[ring_atom]:
                    if i in m_section_predicted["ring_atoms"]:
                        found = True
                        matched_indices.append(i)
                ring_atoms_found.append(found)
            if all(ring_atoms_found):
                correct_atom_rings = True
            if verbose:
                print("test_m:", correct_atom_rings, correct_atom_connector)
            if correct_atom_rings and correct_atom_connector:
                correct = True
                # Consume (multiple fragments can connect to the same ring, so ring atom are not removed)
                gt_to_pred_indices_mapping_m = {
                    k: [
                        idx
                        for idx in v
                        if (idx != m_section_predicted["atom_connector"])
                    ]
                    for k, v in gt_to_pred_indices_mapping_m.items()
                }
                break
        scores["m_sections"].append(correct)

    # Test Sg
    gt_to_pred_indices_mapping_sg = copy.deepcopy(gt_to_pred_indices_mapping)
    gt_sgroups = Chem.rdchem.GetMolSubstanceGroups(gt_molecule)
    predicted_sgroups = Chem.rdchem.GetMolSubstanceGroups(predicted_molecule)

    if verbose:
        for pred_sgroup in predicted_sgroups:
            print(f"predicted_sgroups: {[a for a in pred_sgroup.GetAtoms()]}")
        for gt_sgroup in gt_sgroups:
            print(f"gt_sgroups: {[a for a in gt_sgroup.GetAtoms()]}")
    for gt_sgroup in gt_sgroups:
        force_incorrect = False
        gt_mapped_indices = []
        for i in gt_sgroup.GetAtoms():
            if not (i in gt_to_pred_indices_mapping_sg):
                force_incorrect = True
            gt_mapped_indices.extend(gt_to_pred_indices_mapping_sg[i])
        correct = False
        if verbose:
            print(f"gt_mapped_indices: {gt_mapped_indices}")
        if not (force_incorrect):
            for pred_sgroup in predicted_sgroups:
                # if all(i in gt_mapped_indices for i in pred_sgroup.GetAtoms()) and \
                #     (pred_sgroup.GetProp("LABEL") == gt_sgroup.GetProp("LABEL")):
                if (list(gt_mapped_indices) == list(pred_sgroup.GetAtoms())) and (
                    pred_sgroup.GetProp("LABEL") == gt_sgroup.GetProp("LABEL")
                ):
                    correct = True
                    # Consume
                    gt_to_pred_indices_mapping_m = {
                        k: [idx for idx in v if not (idx in pred_sgroup.GetAtoms())]
                        for k, v in gt_to_pred_indices_mapping_m.items()
                    }
                    break
        scores["sg_sections"].append(correct)

    if len(gt_sgroups) == 0:
        scores["sg_sections"] = None

    # Reduce
    if scores["r_labels"] == None:
        scores["r"] = None
    elif scores["r_labels"] == []:
        scores["r"] = 0.0
    else:
        scores["r"] = np.round(np.mean(scores["r_labels"]), 3)
    if scores["m_sections"] == None:
        scores["m"] = None
    elif scores["m_sections"] == []:
        scores["m"] = 0.0
    else:
        scores["m"] = np.round(np.mean(scores["m_sections"]), 3)
    if scores["sg_sections"] == None:
        scores["sg"] = None
    elif scores["sg_sections"] == []:
        scores["sg"] = 0.0
    else:
        scores["sg"] = np.round(np.mean(scores["sg_sections"]), 3)

    if (
        ((scores["r"] == 1.0) or (scores["r"] == None))
        and ((scores["m"] == 1.0) or (scores["m"] == None))
        and ((scores["sg"] == 1.0) or (scores["sg"] == None))
        and (scores["inchi_equality"] == True)
    ):
        scores["cxsmi_equality"] = True

    scores["valid"] = True
    if verbose:
        pprint(scores)
    return scores
