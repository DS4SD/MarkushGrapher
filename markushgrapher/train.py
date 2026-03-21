#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import datetime
import logging
import os
import warnings

import evaluate
import torch
import transformers
from clearml import Task
from transformers.trainer_utils import is_main_process

import markushgrapher.core.common.begin as begin
from markushgrapher.core.trainers import (
    CurriculumTrainer,
    DataCollator,
    elevateMRCallback,
)
from markushgrapher.utils.common import read_yaml_file
from markushgrapher.utils.model.utils_model_loading import (
    compare_module_weights,
    freeze_module,
    save_weights_separately,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_HOME"] = "/data/.cache/torch"

print("All warnings are ignored!")
warnings.filterwarnings("ignore")


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

    def log_all_args(model_args, data_args, training_args):
        logger.info("===== Model Arguments =====")
        for k, v in vars(model_args).items():
            logger.info(f"{k}: {v}")

        logger.info("===== Data Arguments =====")
        for k, v in vars(data_args).items():
            logger.info(f"{k}: {v}")

        logger.info("===== Training Arguments =====")
        for k, v in vars(training_args).items():
            logger.info(f"{k}: {v}")

    log_all_args(model_args, data_args, training_args)

    if training_args.report_to == "clearml" or training_args.report_to == ["clearml"]:
        logger.info("Initializing CLEARML TASK")
        clearml_task = Task.init(
            project_name="MarkushGrapher-ChemOCR",
            task_name=data_args.clearml_task_name
            + datetime.datetime.now().strftime("%I_%M_%p_%B_%d_%Y"),
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

    if training_args.report_to == "clearml" or training_args.report_to == ["clearml"]:
        clearml_task.connect(model_args, "Custom model args")
        clearml_task.connect(data_args, "Custom data args")
        clearml_task.connect(training_args, "Custom training args")
        clearml_task.connect(dataset_config, "Dataset config")

    # Load model
    device = begin.get_device()
    tokenizer, processor, model = begin.load_markushgrapher(
        model_args,
        data_args,
        training_args,
        device,
        use_pretrained_molscribe=data_args.use_pretrained_molscribe,
    )

    test_correct_loading = False

    if test_correct_loading:
        # OCSR Projector
        ocsr_encoder_is_same = compare_module_weights(
            state_dict=model.encoder.molscribe_encoder.state_dict(),
            weight_stats_path=os.path.dirname(os.path.abspath(__file__))
            + "/../models/submodules/ocsr_encoder/weight_stats/ocsr_encoder_pretrained.json",
        )

        # MLP Projector
        mlp_projector_weights_dir = os.path.dirname(model_args.mlp_projector_weights_filepath)
        submodule = os.path.splitext(os.path.basename(model_args.mlp_projector_weights_filepath))[0]
        mlp_is_same = compare_module_weights(
            state_dict=model.encoder.molscribe_projector.state_dict(),
            weight_stats_path=os.path.join(mlp_projector_weights_dir, f"weight_stats/{submodule}.json"),
        )

        # VTL Encoder
        if model_args.architecture_variant == "me-lf-stack-1":
            vtl_encoder_is_same = compare_module_weights(
                state_dict=model.encoder.molscribe_projector.state_dict(),
                weight_stats_path=os.path.join(mlp_projector_weights_dir, f"weight_stats/{submodule}.json"),
            )

        # VTL Decoder
        vtl_decoder_weights_dir = os.path.dirname(model_args.vtl_decoder_weights_filepath)
        submodule = os.path.splitext(os.path.basename(model_args.vtl_decoder_weights_filepath))[0]
        vtl_decoder_is_same = compare_module_weights(
            state_dict=model.decoder.state_dict(),
            weight_stats_path=os.path.join(vtl_decoder_weights_dir, f"weight_stats/{submodule}.json"),
        )

        print(f"TEST - Correct Loading of OCSR Encoder: {ocsr_encoder_is_same}")
        print(f"TEST - Correct Loading of MLP Projector: {mlp_is_same}")
        print(f"TEST - Correct Loading of VTL Decoder: {vtl_decoder_is_same}")

    # Save submodule weights
    # TODO: fix: Somehow arg is not taken from config, so it is overwritten here
    model_args.save_model_weights_seperately = False
    if model_args.save_model_weights_seperately:
        print("Save Model Weights seperately ...")
        save_weights_separately(
            model=model,
            architecture_variant=model_args.architecture_variant,
            vtl_encoder_save_dir=os.path.dirname(os.path.abspath(__file__))
            + "/../models/submodules/vtl_encoder",
            ocsr_encoder_save_dir=os.path.dirname(os.path.abspath(__file__))
            + "/../models/submodules/ocsr_encoder",
            mlp_projector_save_dir=os.path.dirname(os.path.abspath(__file__))
            + "/../models/submodules/mlp_projector",
            decoder_save_dir=os.path.dirname(os.path.abspath(__file__))
            + "/../models/submodules/vtl_decoder",
            lm_head_save_dir=os.path.dirname(os.path.abspath(__file__))
            + "/../models/submodules/lm_head",
        )

    # Load dataset
    train_dataset = begin.load_dataset(data_args, tokenizer, processor, "train")
    eval_dataset = begin.load_dataset(data_args, tokenizer, processor, "val")

    # Read benchmark dataset
    # uspto-markush - 74 samples
    data_args_benchmark_uspto_markush = copy.deepcopy(data_args)
    data_args_benchmark_uspto_markush.datasets_config = (
        os.path.dirname(__file__)
        + "/../config/datasets/datasets_on_fly_eval_uspto_markush.yaml"
    )
    # wildmol-m - 100 samples
    data_args_benchmark_wildmol_m = copy.deepcopy(data_args)
    data_args_benchmark_wildmol_m.datasets_config = (
        os.path.dirname(__file__)
        + "/../config/datasets/datasets_on_fly_eval_wildmol_m.yaml"
    )
    # uspto (clean) - 50 samples
    data_args_benchmark_uspto = copy.deepcopy(data_args)
    data_args_benchmark_uspto.datasets_config = (
        os.path.dirname(__file__)
        + "/../config/datasets/datasets_on_fly_eval_uspto.yaml"
    )
    # IP5-M - 100 samples
    data_args_benchmark_ip5_m = copy.deepcopy(data_args)
    data_args_benchmark_ip5_m.datasets_config = (
        os.path.dirname(__file__)
        + "/../config/datasets/datasets_on_fly_eval_ip5_m.yaml"
    )

    benchmarks_datasets = {
        "uspto_markush": begin.load_dataset(
            data_args_benchmark_uspto_markush, tokenizer, processor, "test"
        ),
        # Note: Wildmol_M is only reasonable if we do fix_smiles (otherwise incorrect, as it evaluates on expanded abbrevs)
        "wildmol_m": begin.load_dataset(
            data_args_benchmark_wildmol_m, tokenizer, processor, "test"
        ),
        "uspto_clean": begin.load_dataset(
            data_args_benchmark_uspto, tokenizer, processor, "test"
        ),
        "ip5_m": begin.load_dataset(
            data_args_benchmark_ip5_m, tokenizer, processor, "test"
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

    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("After empty_cache()")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

    # Get last checkpoint
    last_checkpoint = begin.last_checkpoint(training_args)
    logger.info(
        """Checkpoint detected, resuming training at %s. To avoid this behavior, change the
                `--output_dir` or add `--overwrite_output_dir` to train from scratch.""",
        last_checkpoint,
    )
    checkpoint = last_checkpoint if last_checkpoint else None
    # Note: resume_from_checkpoint preserves the optimizer but only works if the model name didn't change.
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

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
