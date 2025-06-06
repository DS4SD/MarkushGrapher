#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import logging
import os
import warnings

import evaluate
import transformers
import yaml
from transformers.trainer_utils import is_main_process

import markushgrapher.core.common.begin as begin
from markushgrapher.core.trainers import (CurriculumTrainer, DataCollator,
                                          elevateMRCallback)
from markushgrapher.utils.common import read_yaml_file

print("All warnings are ignored!")
warnings.filterwarnings("ignore")
import datetime

from clearml import Task

os.environ["TORCH_HOME"] = "/data/.cache/torch"


def main():
    # Parse the arguments
    model_args, data_args, training_args = begin.parse_hf_arguments()

    # Set auto parameters
    if training_args.output_dir == "auto":
        training_args.output_dir = "./models/" + datetime.datetime.now().strftime(
            "%I_%M_%p_%B_%d_%Y"
        )
    if model_args.tokenizer_path == "auto":
        model_args.tokenizer_path = model_args.model_name_or_path

    logger = begin.setup_logging(__name__, logging.INFO)

    if training_args.report_to == "clearml":
        clearml_task = Task.init(
            project_name="multimodal-chem-doc",
            task_name=datetime.datetime.now().strftime("%I_%M_%p_%B_%d_%Y"),
        )
    else:
        clearml_task = None

    # Log training details
    if training_args is not None:
        distributed_training = bool(training_args.local_rank != -1)
        logger.warning(
            """Process rank: %s, device: %s, n_gpu: %s distributed training: %s,
                       16-bits training: %s""",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            distributed_training,
            training_args.fp16,
        )

        # Set the verbosity to info of the Transformers logger (on main process only):
        if is_main_process(training_args.local_rank):
            transformers.utils.logging.set_verbosity_info()
            transformers.utils.logging.enable_default_handler()
            transformers.utils.logging.enable_explicit_format()

        logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Model arguments: {model_args}")

    dataset_config = read_yaml_file(data_args.datasets_config)

    if training_args.report_to == "clearml":
        clearml_task.connect(model_args, "Custom model args")
        clearml_task.connect(data_args, "Custom data args")
        clearml_task.connect(training_args, "Custom training args")
        clearml_task.connect(dataset_config, "Dataset config")

    # Load model
    device = begin.get_device()
    tokenizer, processor, model = begin.load_markushgrapher(
        model_args, data_args, training_args, device
    )

    # Load dataset
    train_dataset = begin.load_dataset(data_args, tokenizer, processor, "train")
    eval_dataset = begin.load_dataset(data_args, tokenizer, processor, "val")
    test_dataset = begin.load_dataset(data_args, tokenizer, processor, "test")

    # Read benchmark dataset
    data_args_benchmark_lum_test = copy.deepcopy(data_args)
    data_args_benchmark_lum_test.datasets_config = (
        os.path.dirname(__file__)
        + f"/../config/datasets/datasets_on_fly_eval_lum_test.yaml"
    )
    data_args_benchmark_uspto_markush = copy.deepcopy(data_args)
    data_args_benchmark_uspto_markush.datasets_config = (
        os.path.dirname(__file__)
        + f"/../config/datasets/datasets_on_fly_eval_uspto_markush.yaml"
    )
    benchmarks_datasets = {
        "lum_test": begin.load_dataset(
            data_args_benchmark_lum_test, tokenizer, processor, "test"
        ),
        "uspto_markush": begin.load_dataset(
            data_args_benchmark_uspto_markush, tokenizer, processor, "test"
        ),
    }

    # Data collator
    padding = "max_length" if data_args.pad_to_max_length else False
    data_collator = DataCollator(
        tokenizer=tokenizer,
        padding=padding,
        max_length=data_args.max_seq_length,
        max_length_decoder=data_args.max_seq_length_decoder,
    )

    # Initialize Trainer
    def preprocess_logits_for_metrics(logits, labels):
        return logits[0].argmax(dim=-1)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        metric = evaluate.load("accuracy")
        return metric.compute(predictions=predictions, references=labels)

    elevateMRcallback = elevateMRCallback(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        early_stopping_patience=data_args.curri_patience,
        early_stopping_threshold=data_args.curri_threshold,
    )

    trainer = CurriculumTrainer(
        model=model,
        data_args=data_args,
        model_args=model_args,
        clearml_task=clearml_task,
        training_args=training_args,
        benchmarks_datasets=benchmarks_datasets,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[elevateMRcallback] if data_args.curriculum else None,
        data_collator=data_collator,
        loss_fct=model_args.loss_fct,
    )

    # Get last checkpoint
    last_checkpoint = begin.last_checkpoint(training_args)
    logger.info(
        """Checkpoint detected, resuming training at %s. To avoid this behavior, change the
                `--output_dir` or add `--overwrite_output_dir` to train from scratch.""",
        last_checkpoint,
    )
    checkpoint = last_checkpoint if last_checkpoint else None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    # Note: resume_from_checkpoint() should preserve the optimizer but currently works only if the model name did not change.
    # (The loading of pretrained weights is handled by from_pretrained() anyway.)

    metrics = train_result.metrics
    trainer.save_model()  # Saves the tokenizer too for easy upload

    max_train_samples = (
        data_args.max_train_samples
        if data_args.max_train_samples is not None
        else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
