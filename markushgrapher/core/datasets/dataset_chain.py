import importlib
import logging
from ast import literal_eval

import torch
from datasets import arrow_dataset
from torch.utils.data import ConcatDataset, Dataset

from markushgrapher.core.datasets.task_collator import TaskCollator
from markushgrapher.utils.common import read_yaml_file

logger = logging.getLogger(__name__)


class DatasetChain:
    def __init__(
        self,
        processors,
        tokenizer,
        data_args,
        split="train",
    ):
        logger.info("Loading dataset dictionnary")

        # Encoding specific members
        self._data_args = data_args
        self._unit = data_args.unit
        self._processors = processors
        self._processor_no_ocr = processors["no_ocr"]
        self._tokenizer = tokenizer
        self._image_size = data_args.image_size
        self._split = split

        # Initialize the collator
        self._collator = TaskCollator(tokenizer)

        # Load the datasets
        self._datasets = self.get_datasets(data_args)

        if len(self._datasets) == 1 or (self._data_args.mode == "loaders"):
            self._all_datasets = self._datasets

    def get_datasets(self, data_args):
        # Loads all datasets inside datasets_config by class
        datasets = {}
        dataset_configs = read_yaml_file(data_args.datasets_config)
        for _, dataset_config in dataset_configs.items():
            module_name = dataset_config["module_name"]
            class_name = dataset_config["class_name"]
            name = dataset_config["name"]
            print(module_name, class_name, name, dataset_config["dataset_path"])
            module = importlib.import_module(
                "markushgrapher.core.datasets." + module_name
            )
            dataset_class = getattr(module, class_name)
            dataset_instance = dataset_class(
                dataset_config,
                self._data_args,
                self._tokenizer,
                self._processors,
                self._collator,
                self._split,
            )
            datasets[name] = dataset_instance
        return datasets

    def __len__(self):
        return len(self._all_datasets)
