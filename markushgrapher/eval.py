#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import transformers
import yaml
from markushgenerator.cxsmiles_tokenizer import CXSMILESTokenizer

import markushgrapher.core.common.begin as begin
from markushgrapher.core.common.markush_tokenizer import MarkushTokenizer
from markushgrapher.utils.common import read_yaml_file
from markushgrapher.utils.ocsr.utils_evaluation import get_smiles_metrics


def main():
    # Parse the arguments
    model_args, data_args, training_args = begin.parse_hf_arguments()

    # Get the logger
    logger = begin.setup_logging(__name__, training_args.log_level)

    max_eval_samples = 3
    display_eval_samples = True
    max_display_eval_samples = 300
    display_markush_evaluation = True
    display_errors = True
    verbose = True
    get_training_smiles = False
    read_training_smiles = True
    overwrite_training_smiles = True
    read_predictions = False
    overwrite_predictions = True
    selected_indices = []
    if selected_indices != []:
        max_eval_samples = max(selected_indices) + 1
    input_encoding_training_dataset = "mdu_3005"  # None, "mdu_3005"
    # Note: "input_encoding_training_dataset" configures the input encoding/decoding
    # dataset_config["training_dataset_name"] configures the prediction encoding/decoding

    # Log
    if training_args is not None:
        logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Model arguments: {model_args}")

    # Load model
    device = begin.get_device()
    tokenizer, processors, model = begin.load_markushgrapher(
        model_args, data_args, training_args, device
    )

    # Load tokenizer
    # Note: Having two separate tokenizer can be useful if the vocabulary for testing and traning are different.
    # 'markush_tokenizer' encodes inputs, 'markush_tokenizer_training' encodes predictions. (An alternative would be encode new tokens as <unknown>)
    dataset_config = list(read_yaml_file(data_args.datasets_config).values())[0]
    markush_tokenizer = MarkushTokenizer(
        tokenizer,
        dataset_config["dataset_path"],
        encode_position=dataset_config["encode_position"],
        grounded_smiles=dataset_config["grounded_smiles"],
        encode_index=dataset_config["encode_index"],
        training_dataset_name=input_encoding_training_dataset,
    )
    markush_tokenizer_training = MarkushTokenizer(
        tokenizer,
        dataset_config["dataset_path"],
        encode_position=dataset_config["encode_position"],
        grounded_smiles=dataset_config["grounded_smiles"],
        encode_index=dataset_config["encode_index"],
        training_dataset_name=dataset_config["training_dataset_name"],
    )

    # Load dataset
    split = "test"
    dataset_dict = begin.load_dataset(data_args, tokenizer, processors, split)
    dataset_name = "mdu"
    dataset = dataset_dict[dataset_name]
    # dataset._all_datasets = list(dataset._datasets.values())[0] # MODIFIES EVREYTHING!!
    logger.info(
        f"Dataset '{dataset_name}' max index: {min(len(dataset), max_eval_samples)}"
    )

    # Get smiles in the training set
    cxsmiles_tokenizer = CXSMILESTokenizer(
        training_dataset=input_encoding_training_dataset,
        condense_labels=dataset_config["condense_labels"],
    )
    cxsmiles_tokenizer_training = CXSMILESTokenizer(
        training_dataset=dataset_config["training_dataset_name"],
        condense_labels=dataset_config["condense_labels"],
    )
    if get_training_smiles:
        training_smiles = get_training_smiles(
            dataset_config["dataset_path"],
            cxsmiles_tokenizer,
            read_training_smiles=read_training_smiles,
            overwrite_training_smiles=overwrite_training_smiles,
            verbose=verbose,
        )
    else:
        training_smiles = []

    if data_args.viz_out_dir == "auto":
        data_args.viz_out_dir = (
            "./data/visualization/prediction/"
            + dataset_config["dataset_path"].split("/")[-1]
            + "/"
            + "-".join(model_args.model_name_or_path.split("/")[-2:])
        )
    if training_args.output_dir == "auto":
        training_args.output_dir = (
            "./data/evaluation/"
            + dataset_config["dataset_path"].split("/")[-1]
            + "/"
            + "-".join(model_args.model_name_or_path.split("/")[-2:])
        )

    if not (os.path.exists(data_args.viz_out_dir)):
        os.makedirs(data_args.viz_out_dir, exist_ok=True)
    os.makedirs(training_args.output_dir, exist_ok=True)

    metrics = get_smiles_metrics(
        model,
        dataset=dataset,
        max_eval_samples=max_eval_samples,
        tokenizer=tokenizer,
        training_smiles=training_smiles,
        markush_tokenizer=markush_tokenizer,
        cxsmiles_tokenizer=cxsmiles_tokenizer,
        display_eval_samples=display_eval_samples,
        device=device,
        display_samples_output_dir=data_args.viz_out_dir,
        training_args=training_args,
        model_args=model_args,
        config=dataset_config,
        selected_indices=selected_indices,
        read_predictions=read_predictions,
        overwrite_predictions=overwrite_predictions,
        save_scores=True,
        display_markush_evaluation=display_markush_evaluation,
        display_errors=display_errors,
        max_display_eval_samples=max_display_eval_samples,
        verbose=verbose,
        markush_tokenizer_training=markush_tokenizer_training,
        cxsmiles_tokenizer_training=cxsmiles_tokenizer_training,
    )
    print(metrics)


if __name__ == "__main__":
    main()
