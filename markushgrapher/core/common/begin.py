import logging
import os
import sys

import torch
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.models.markushgrapher import (
    MarkushgrapherConfig,
    MarkushgrapherForConditionalGeneration,
    MarkushgrapherImageProcessor,
    MarkushgrapherProcessor,
    MarkushgrapherTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

from markushgrapher.core.common.arguments import DataTrainingArguments, ModelArguments
from markushgrapher.core.datasets.dataset_chain import DatasetChain
from markushgrapher.utils.model.utils_model_loading import freeze_module

def setup_logging(logger_name: str, loglevel=logging.INFO):
    logger = logging.getLogger(logger_name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(loglevel)
    return logger


def parse_hf_arguments() -> (
    tuple[ModelArguments, DataTrainingArguments, TrainingArguments]
):
    check_min_version("4.6.0")

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_yaml_file(
        yaml_file=os.path.abspath(sys.argv[1])
    )

    training_args.logging_dir = os.path.join(training_args.output_dir, "runs")
    if model_args.cache_dir is None:
        model_args.cache_dir = os.path.join(training_args.output_dir, "cache")

    if training_args.do_train:
        os.makedirs(model_args.cache_dir, exist_ok=True)

    if model_args.tokenizer_path == "auto":
        if model_args.model_name_or_path != "":
            model_args.tokenizer_path = model_args.model_name_or_path
        elif model_args.config_name != "":
            model_args.tokenizer_path = model_args.config_name

    return model_args, data_args, training_args


def last_checkpoint(training_args: TrainingArguments) -> str:
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
    return last_checkpoint


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def load_markushgrapher(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    device,
    tokenizer_pretrained="",
    model_config_pretrained="",
    use_pretrained_molscribe=True,
    ignore_model=False,
) -> tuple[
    MarkushgrapherTokenizer,
    MarkushgrapherProcessor,
    MarkushgrapherForConditionalGeneration,
]:
    r"""
    Load the Markushgrapher model.
    The tokenizer and model config are loaded from their pretrained counterparts.
    """
    set_seed(training_args.seed)

    image_size = {"height": data_args.image_size, "width": data_args.image_size}
    image_processor_no_ocr = MarkushgrapherImageProcessor(
        apply_ocr=False,
        size=image_size,
    )

    tokenizer = MarkushgrapherTokenizer.from_pretrained(model_args.tokenizer_path)

    # Config
    if model_args.model_name_or_path != "":
        config = MarkushgrapherConfig.from_pretrained(model_args.model_name_or_path)
    elif model_args.config_name != "":
        config = MarkushgrapherConfig.from_pretrained(model_args.config_name)

    config.image_size = data_args.image_size
    config.architecture_variant = model_args.architecture_variant
    config.output_attentions = True

    # Model
    if ignore_model:
        model = None
    else:
        if model_args.model_name_or_path == "":
            model = MarkushgrapherForConditionalGeneration(config).to(device)
        else:
            model = MarkushgrapherForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                config=config,
            ).to(device)

        print(f"Use Pretrained Molscribe weights: {use_pretrained_molscribe}")

        if use_pretrained_molscribe:
            model.init_molscribe_weights()

        # Load submodule weights (e.g. from previous Markushgrapher training runs)
        if model_args.load_submodule_weights:
            print("Load submodule weights ...")

            # Load MLP Projector
            if model_args.mlp_projector_weights_filepath:
                if os.path.exists(model_args.mlp_projector_weights_filepath):
                    print(f"Load MLP Projector weights: {model_args.mlp_projector_weights_filepath} ...")
                    projector_states = torch.load(
                        model_args.mlp_projector_weights_filepath, map_location="cpu"
                    )
                    model.safe_load(model.encoder.molscribe_projector, projector_states)

                    if model_args.freeze_mlp_projector:
                        freeze_module(model.encoder.molscribe_projector)
                        print("MLP Projector has been frozen.")
                else:
                    print(f"MLP Projector weights not found at {model_args.mlp_projector_weights_filepath}")

            # Load VTL Decoder
            if model_args.vtl_decoder_weights_filepath:
                if os.path.exists(model_args.vtl_decoder_weights_filepath):
                    print(f"Load VTL Decoder weights: {model_args.vtl_decoder_weights_filepath} ...")
                    decoder_states = torch.load(
                        model_args.vtl_decoder_weights_filepath, map_location="cpu"
                    )
                    model.safe_load(model.decoder, decoder_states)

                    if model_args.freeze_vtl_decoder:
                        freeze_module(model.decoder)
                        print("VTL Decoder has been frozen.")
                else:
                    print(f"VTL Decoder weights not found at {model_args.vtl_decoder_weights_filepath}")

    processors = {
        "no_ocr": MarkushgrapherProcessor(
            image_processor=image_processor_no_ocr, tokenizer=tokenizer
        )
    }
    return tokenizer, processors, model

def load_dataset(
    data_args: DataTrainingArguments,
    tokenizer: MarkushgrapherTokenizer,
    processors: MarkushgrapherProcessor,
    split: str,
) -> DatasetChain:
    dataset_dict = DatasetChain(
        processors=processors,
        tokenizer=tokenizer,
        data_args=data_args,
        split=split,
    )._all_datasets
    return dataset_dict
