import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from markushgenerator.cxsmiles_tokenizer import CXSMILESTokenizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torchvision.ops import generalized_box_iou_loss
from tqdm import tqdm
from transformers import Trainer
from transformers.deepspeed import deepspeed_init
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import (IterableDatasetShard,
                                           find_batch_size, nested_concat,
                                           nested_numpify, nested_truncate)
from transformers.trainer_utils import (EvalLoopOutput, EvalPrediction,
                                        IntervalStrategy,
                                        denumpify_detensorize, has_length,
                                        seed_worker)

import markushgrapher.core.common.begin as begin
from markushgrapher.core.common.markush_tokenizer import MarkushTokenizer
from markushgrapher.utils.ocsr.utils_evaluation import get_smiles_metrics
from markushgrapher.utils.ocsr.utils_training import get_training_smiles

from ..common.utils import calculate_iou
from . import losses

logger = logging.getLogger(__name__)


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """

    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """

    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        if self.dataset._config["stream"]:
            if self.dataset._split == "train":
                return int(self.dataset._config["samples"] * 0.9)
            else:
                return int(self.dataset._config["samples"] * 0.1)
        else:
            return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }

        self.task_name_list = list(self.dataloader_dict)

        self.dataset = [None] * sum(
            dataloader.__len__() for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.
        """
        num_tasks = len(self.task_name_list)

        probabilities = (
            np.ones(num_tasks) / num_tasks
        )  # Equal probability for all tasks

        task_choice_list = np.random.choice(
            num_tasks,
            size=sum(self.num_batches_dict.values()),
            p=probabilities,
            replace=True,
        )

        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }

        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            try:
                yield next(dataloader_iter_dict[task_name])
            except StopIteration:
                # Recreate the dataloader iterator
                dataloader_iter_dict[task_name] = iter(self.dataloader_dict[task_name])
                yield next(dataloader_iter_dict[task_name])


class CurriculumTrainer(Trainer):
    def __init__(
        self,
        data_args,
        model_args,
        clearml_task,
        training_args,
        benchmarks_datasets,
        *args,
        **kwargs,
    ):
        self.lm_ratio = None
        self.loss_fct = None
        if "loss_fct" in kwargs:
            self.loss_fct = kwargs.pop("loss_fct")
        super().__init__(*args, **kwargs)

        self.device = begin.get_device()
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args

        self.__dict__.update(kwargs)

        with open(data_args.datasets_config) as stream:
            yaml_file = yaml.safe_load(stream)

        self.cxsmiles_tokenizer = CXSMILESTokenizer(
            condense_labels=list(yaml_file.values())[0]["condense_labels"]
        )
        self.markush_tokenizer = MarkushTokenizer(
            self.tokenizer,
            list(yaml_file.values())[0]["dataset_path"],
            encode_position=list(yaml_file.values())[0]["encode_position"],
            grounded_smiles=list(yaml_file.values())[0]["grounded_smiles"],
            encode_index=list(yaml_file.values())[0]["encode_index"],
            condense_labels=list(yaml_file.values())[0]["condense_labels"],
        )

        self.training_smiles = []
        # self.training_smiles = get_training_smiles(list(yaml_file.values())[0]["dataset_path"], self.cxsmiles_tokenizer, read_training_smiles=True, overwrite_training_smiles=True, verbose=False)

        self.clearml_task = clearml_task
        self.global_eval_step = 0

        # Create debug samples folder
        p = (
            os.path.dirname(__file__)
            + f"/../../../data/visualization/training_debug/{training_args.output_dir.split('/')[-1]}"
        )
        if not (os.path.exists(p)):
            os.mkdir(p)

        self.benchmarks_datasets = benchmarks_datasets

    def log(self, logs: Dict[str, float]) -> None:
        # """
        # Log `logs` on the various objects watching training.

        # Subclass and override this method to inject custom behavior.

        # Args:
        #     logs (`Dict[str, float]`):
        #         The values to log.
        # """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # For backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = (
            self.args.data_seed if self.args.data_seed is not None else self.args.seed
        )
        if self.args.world_size <= 1:
            if train_dataset._config["stream"]:
                data_loader = DataLoaderWithTaskname(
                    task_name=task_name,
                    data_loader=DataLoader(
                        train_dataset,
                        batch_size=self.args.train_batch_size,
                        collate_fn=self.data_collator,
                    ),
                )
                return data_loader
            else:
                train_sampler = RandomSampler(train_dataset)
                data_loader = DataLoaderWithTaskname(
                    task_name=task_name,
                    data_loader=DataLoader(
                        train_dataset,
                        batch_size=self.args.train_batch_size,
                        sampler=train_sampler,
                        collate_fn=self.data_collator,
                    ),
                )
        else:
            if train_dataset._config["stream"]:
                data_loader = DataLoaderWithTaskname(
                    task_name=task_name,
                    data_loader=DataLoader(
                        train_dataset,
                        batch_size=self.args.train_batch_size,
                        collate_fn=self.data_collator,
                    ),
                )
            else:
                train_sampler = DistributedSampler(train_dataset, seed=seed)
                data_loader = DataLoaderWithTaskname(
                    task_name=task_name,
                    data_loader=DataLoader(
                        train_dataset,
                        batch_size=self.args.train_batch_size,
                        sampler=train_sampler,
                        collate_fn=self.data_collator,
                    ),
                )

        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        if len(self.train_dataset) == 1:
            task_name, task_dataset = next(iter(self.train_dataset.items()))
            return self.get_single_train_dataloader(task_name, task_dataset)
        else:
            return MultitaskDataloader(
                {
                    task_name: self.get_single_train_dataloader(task_name, task_dataset)
                    for task_name, task_dataset in self.train_dataset.items()
                }
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_fct == "CE":
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            return (loss, outputs) if return_outputs else loss

        else:
            logits = model(**inputs).logits
            labels = inputs.get("labels").to(logits.device)

            last_true_label_index = (labels == 1).nonzero(as_tuple=False).max()

            ce_loss = 0  # Cross Entropy loss
            loc_loss = 0  # Location token loss

            # Mask for elements if <loc_{idx}> in range [33000, 32500] inclusive
            max_logits_indices = torch.argmax(logits, dim=2)
            mask_loc = ((labels >= 32500) & (labels <= 33000)) & (
                (max_logits_indices >= 32500) & (max_logits_indices <= 33000)
            )

            # Validation for when some tokens are not locations tokens and when boxes are rotated
            for idx in range(1, last_true_label_index, 5):
                if (
                    not torch.all(mask_loc[0, idx : idx + 4])
                    or (max_logits_indices[0, idx] < max_logits_indices[0, idx + 2])
                    or (max_logits_indices[0, idx + 1] < max_logits_indices[0, idx + 3])
                ):
                    mask_loc[0, idx : idx + 4] = False

            ce_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            ce_loss = ce_loss_fct(
                logits[~mask_loc].view(-1, logits.size(-1)), labels[~mask_loc].view(-1)
            )

            # If there are any location tokens in the batch, compute location loss
            if torch.any(mask_loc):
                input = (33000 - max_logits_indices[mask_loc]) / 500
                target = (33000 - labels[mask_loc]) / 500

                if self.loss_fct == "Huber":
                    loc_loss = F.smooth_l1_loss(input, target)
                elif self.loss_fct == "MSE":
                    loc_loss = F.mse_loss(input, target)
                elif self.loss_fct == "Custom_huber":
                    loc_loss = losses.custom_huber2(input, target, 2)
                elif self.loss_fct == "GIOU":
                    input = torch.reshape(input, (-1, 4))
                    target = torch.reshape(target, (-1, 4))
                    loc_loss = generalized_box_iou_loss(input, target, "mean")

            loss = ce_loss + loc_loss

            return (loss, logits) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):  # -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.model_wrapped is self.model:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # For the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # Backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # If full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)

            if step > self.training_args.max_eval_samples // observed_batch_size:
                break

            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            inputs_decode = (
                self._prepare_input(inputs["input_ids"])
                if args.include_inputs_for_metrics
                else None
            )

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = (
                    losses
                    if losses_host is None
                    else torch.cat((losses_host, losses), dim=0)
                )
            if labels is not None:
                labels = self._pad_across_processes(labels)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self._nested_gather(logits)
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=-100)
                )
            if labels is not None:
                labels = self._nested_gather(labels)
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=-100)
                )
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = (
                        losses
                        if all_losses is None
                        else np.concatenate((all_losses, losses), axis=0)
                    )
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = (
                        logits
                        if all_preds is None
                        else nested_concat(all_preds, logits, padding_index=-100)
                    )
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(
                            all_inputs, inputs_decode, padding_index=-100
                        )
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels
                        if all_labels is None
                        else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = (
                    None,
                    None,
                    None,
                    None,
                )

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = (
                losses
                if all_losses is None
                else np.concatenate((all_losses, losses), axis=0)
            )
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = (
                logits
                if all_preds is None
                else nested_concat(all_preds, logits, padding_index=-100)
            )
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode
                if all_inputs is None
                else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = (
                labels
                if all_labels is None
                else nested_concat(all_labels, labels, padding_index=-100)
            )

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif (
            isinstance(eval_dataset, IterableDatasetShard)
            and getattr(eval_dataset, "num_examples", 0) > 0
        ):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # Samplers have been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
        ):
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(
                        predictions=all_preds, label_ids=all_labels, inputs=all_inputs
                    )
                )
            else:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels)
                )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = (
                self.jit_compilation_time
            )

        print("Starting evaluation synthetic")
        custom_metrics = self.compute_custom_metrics(model, eval_dataset)
        print(f"Evaluation metrics: {custom_metrics}")

        metrics.update(custom_metrics)

        for benchmark in ["lum_test", "uspto_markush"]:
            print(f"Starting benchmark evaluation {benchmark}")
            custom_metrics_benchmark = self.compute_custom_metrics(
                model,
                self.benchmarks_datasets[benchmark]["mdu"],
                metrics_prefix=f"{benchmark}_",
            )
            print(f"Benchmark evaluation metrics: {custom_metrics_benchmark_lum_test}")
            metrics.update(custom_metrics_benchmark)

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )

    def compute_custom_metrics(
        self, model, dataset=None, verbose=False, metrics_prefix=""
    ):
        print(f"Compute custom metrics on: {dataset._config['name']}")
        metrics = {"accuracy": 0, "mae": 0, "iou": 0}

        correct_predictions = 0
        total_predictions = 0
        mae_sum = 0
        mae_count = 0
        iou_sum = 0
        iou_count = 0

        print("--- Starting evaluation ---")
        for i in tqdm(range(min(len(dataset), self.training_args.max_eval_samples))):
            sample = self.data_collator([dataset[i]])

            for key, value in sample.items():
                if torch.is_tensor(value):
                    sample[key] = value.to(self.model.device)

            with torch.no_grad():
                logits = model(**sample).logits
                pred = torch.argmax(logits, dim=2)
                label = sample["labels"].to(self.model.device)

            # Find the index of the last true label (where label equals 1, 1 being the end of sentence token)
            non_zero_indices = (label == 1).nonzero(as_tuple=False)
            if non_zero_indices.numel() > 0:
                last_true_label_index = non_zero_indices.max()
            else:
                print("Stop label (1) not found in the ground-truth")
                last_true_label_index = label.size(1)

            # Slice the pred and label tensors up to the last true label index
            pred_sliced = pred[:, : last_true_label_index + 1]
            label_sliced = label[:, : last_true_label_index + 1]

            # Calculate accuracy for this batch
            correct_predictions += torch.sum(pred_sliced == label_sliced).item()
            total_predictions += pred_sliced.numel()

            # Apply mask for <loc> tokens
            mask_mse = ((label_sliced >= 32500) & (label_sliced <= 33000)) & (
                (pred_sliced >= 32500) & (pred_sliced <= 33000)
            )

            if torch.any(mask_mse):
                mae_sum += (
                    torch.abs(label_sliced[mask_mse] - pred_sliced[mask_mse])
                    .sum()
                    .item()
                )
                mae_count += mask_mse.sum().item()

            # Calculate IOU for <loc> tokens
            for idx in range(mask_mse.size(1) - 3):
                if torch.all(
                    mask_mse[0, idx : idx + 4]
                ):  # Check for four consecutive True values
                    pred_box = [
                        self.tokenizer.decode(token_id)
                        for token_id in pred_sliced[0, idx : idx + 4]
                    ]  # Extract 4 tokens for the bounding box
                    label_box = [
                        self.tokenizer.decode(token_id)
                        for token_id in label_sliced[0, idx : idx + 4]
                    ]

                    iou_sum += calculate_iou(pred_box, label_box)
                    iou_count += 1

        # Calculate metrics
        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )
        mae = mae_sum / mae_count if mae_count > 0 else 0
        iou = iou_sum / iou_count if iou_count > 0 else 0

        metrics["accuracy"] = accuracy
        metrics["mae"] = mae
        metrics["iou"] = iou

        print(
            f"Starting get_smiles_metrics with max_eval_samples: {self.training_args.max_eval_samples}"
        )
        smiles_metrics = get_smiles_metrics(
            model,
            dataset,
            max_eval_samples=self.training_args.max_eval_samples,
            tokenizer=self.tokenizer,
            training_smiles=self.training_smiles,
            markush_tokenizer=self.markush_tokenizer,
            cxsmiles_tokenizer=self.cxsmiles_tokenizer,
            display_eval_samples=False,
            device=self.device,
            display_samples_output_dir=os.path.dirname(__file__)
            + f"/../../../data/visualization/training_debug/{self.training_args.output_dir.split('/')[-1]}/{self.global_eval_step}/",
            training_args=self.training_args,
            config=dataset._config,
            read_predictions=False,
            overwrite_predictions=False,
            save_scores=True,
            display_markush_evaluation=False,
            display_errors=True,
            max_display_eval_samples=20,
            verbose=False,
            markush_tokenizer_training=None,
            metrics_prefix=metrics_prefix,
        )
        metrics.update(smiles_metrics)

        self.global_eval_step += 1
        return metrics


class elevateMRCallback(TrainerCallback):
    def __init__(
        self,
        train_dataset,
        eval_dataset,
        early_stopping_patience: int = 1,
        early_stopping_threshold: Optional[float] = 0.0,
    ):
        self.train_d = train_dataset
        self.eval_d = eval_dataset
        self.min_loss_per_MR = None  # arbitrary MAX loss value
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def on_step_end(self, args, state, control, **kwargs):
        return

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.less
        if self.min_loss_per_MR is None or (
            operator(metric_value, self.min_loss_per_MR)
            and abs(metric_value - self.min_loss_per_MR) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
            self.min_loss_per_MR = metric_value
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        assert (
            args.evaluation_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = "loss"
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
