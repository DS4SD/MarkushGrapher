from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(
        default="ner", metadata={"help": "The name of the task (ner, pos...)."}
    )
    unit: Optional[str] = field(
        default="word", metadata={"help": "The unit of tokenize (word, token)."}
    )
    curriculum: Optional[str] = field(
        default=False,
        metadata={"help": "The choice of curriculum learning (True or False)."},
    )
    curri_patience: Optional[int] = field(
        default=None, metadata={"help": "Number of times it was not been updated"}
    )
    curri_threshold: Optional[int] = field(
        default=None,
        metadata={"help": "Criteria for determining that an update has been made"},
    )
    curri_start_MR: Optional[int] = field(
        default=None,
        metadata={
            "help": "The starting point of masking ratio from curri_start_MR to 100%"
        },
    )

    data_type: Optional[str] = field(
        default=None,
        metadata={"help": "data type json or HF"},
    )

    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "local dataset stored location"},
    )

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or JSON file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to predict on (a csv or JSON file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length"
            " in the batch. More efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training "
            "examples to this value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation "
            "examples to this value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test "
            "examples to this value if set."
        },
    )
    image_size: Optional[int] = field(
        default=512,
        metadata={"help": "image size" "value if set."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_seq_length_decoder: int = field(
        default=16,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    do_save_visualize: bool = field(
        default=False,
        metadata={"help": "Whether to save visualizations in predict"},
    )
    viz_out_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to save the output visualization from predic."},
    )
    apply_ocr: bool = field(
        default=False,
        metadata={"help": "Whether to apply tesseract OCR or cells"},
    )
    datasets_config: Optional[str] = field(
        default=None,
        metadata={"help": "local dataset stored location"},
    )

    max_eval_samples: int = field(
        default=1000,
        metadata={"help": "The number of samples to use for the training on-fly evaluation."},
    )


@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to train from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    tokenizer_path: str = field(
        default=None,
        metadata={"help": "Path to tokenizer from huggingface.co/models"},
    )
    model_type: str = field(
        default=None, metadata={"help": "Model type selected in the list."}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name, commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` "
            "(necessary to use this script with private models)."
        },
    )
    attention_type: str = field(
        default="original_full",
        metadata={
            "help": "Attention type: BigBird configuration only. Choices: block_sparse (default) "
            "or original_full"
        },
    )
    loss_fct: str = field(
        default="CE",
        metadata={"help": "Loss function for location tokens. Default: None"},
    )

    architecture_variant: str = field(
        default='none',
        metadata={"help": "The MarkushGrapher architecture variant."},
    )
    
    beam_search: bool = field(
        default=False,
        metadata={"help": "Whether to use beam search."},
    )
    
    normalize_bbox: bool = field(
        default=False,
        metadata={"help": "Whether to normalize OCR boxes to [0, 1]."},
    )
