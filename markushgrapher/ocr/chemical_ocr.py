import re
import time

import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def parse_ocr_string(ocr_string: str):
    """Convert predicted string to list of dictionaries."""
    # Step 1: Remove <ocr> tags and the leading fixed <loc_*> tags
    cleaned = re.sub(r"</?ocr>", "", ocr_string).strip()
    cleaned = re.sub(r"^<loc_0><loc_0><loc_500><loc_500>", "", cleaned, count=1).strip()

    # Step 2: Split into lines
    lines = cleaned.splitlines()

    # Step 3: Extract bbox + text from each line
    words = []
    normalized_boxes = []

    for line in lines:
        # Use raw strings for regex to avoid escape warnings
        locs = list(map(int, re.findall(r"<loc_(\d+)>", line)))
        text = re.sub(r"(?:<loc_\d+>){4}", "", line).strip()
        if len(locs) >= 4 and text:
            bbox = locs[-4:]
            bbx_conv = [x / 500 for x in bbox]

            words.append(text)
            normalized_boxes.append(bbx_conv)

    return words, normalized_boxes


def clean_ocr_text(text, start_tag="<ocr>", end_tag="</ocr>"):
    """
    Removes everything before the `start_tag` and after the `end_tag` if provided.

    Args:
        text (str): Input text.
        start_tag (str): The tag after which content is kept.
        end_tag (str or None): The tag before which content is kept.

    Returns:
        str: Cleaned text.
    """
    # Remove everything before the start_tag
    pattern_start = rf"^.*?({re.escape(start_tag)})"
    text = re.sub(pattern_start, r"\1", text, flags=re.DOTALL)

    # Remove everything after the end_tag (if provided)
    if end_tag:
        pattern_end = rf"({re.escape(end_tag)}).*?$"
        text = re.sub(pattern_end, r"\1", text, flags=re.DOTALL)

    return text


class Chemical_OCR:
    def __init__(
        self,
        model_path: str = "checkpoints/chemicalocr_v3/checkpoint-1768",
        batch_size: int = 8192,
        log_interval: int = 100,
    ):
        """Initialize the OCR model.

        Args:
            model_path (str): Path to model checkpoint.
            batch_size (int): Batch size for inference.
            log_interval (int): Logging interval.
        """
        self.model_path = model_path
        self.llm = LLM(model=self.model_path, limit_mm_per_prompt={"image": 1})
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.batch_size = batch_size
        self.log_interval = log_interval

        if torch.cuda.is_available():
            print(
                f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
            print(
                f"Allocated GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
            )
            print(f"Cached GPU Memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    def load_hf_dataset(self, hf_dataset_dir: str):
        """
        Load huggingface dataset
        """
        print("Load Test set...")
        test_dataset = load_from_disk(hf_dataset_dir)

        pil_images = test_dataset["page_image"]
        image_names = test_dataset["image_name"]

        return pil_images, image_names

    def prepare_prompt(self, prompt="Perform OCR on this chemical structure image."):
        """Prepare prompt."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    @staticmethod
    def replace_cells(sample, name_to_cells):

        # for markush-synthetic-training
        #page_image_path = sample["page_image_path"]  
        #img_name = os.path.basename(page_image_path)[:-4]

        # for m2s
        #img_name = sample["image_name"] # m2s

        # for ip5_m
        img_name = sample["id"] 

        if img_name in name_to_cells:
            sample["cells"] = name_to_cells[img_name]
        else:
            sample["cells"] = []
        return sample

    def predict(self, dataset_dir: str, output_dir: str, max_len=8192, split="train", postprocess=True, verbose=False):
        """
        Perform OCR on a given huggingface image dataset

        Args:
            dataset_dir (str): Path to hf dataset (later also possible: dir of images)
            output_dir (str): Output path of new dataset (model predicted OCR cells in dataset["cells"])
            max_len (int): max length of predicted output (?)
        """

        OVERLAP = 0.3

        # Load hf dataset
        print("Load Test set...")
        dataset = load_from_disk(dataset_dir)
        pil_images = dataset["page_image"]
        #image_names = dataset['image_name'] # m2s
        image_names = dataset["id"] # ip5_m

        # markush-synthetic
        #page_image_paths = dataset["page_image_path"] 
        #image_names = [os.path.basename(file_path)[:-4] for file_path in page_image_paths]

        # Prepare Prompt
        prompt = self.prepare_prompt()

        # Sampling params
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096,
        )

        start_time = time.time()
        name_to_cells = {}

        for i in range(0, len(pil_images), self.batch_size):

            # Load image and image name
            batch_images = pil_images[i : i + self.batch_size]
            batch_names = image_names[i : i + self.batch_size]

            load_start_time = time.time()

            llm_inputs = []
            for image in batch_images:
                llm_inputs.append(
                    {"prompt": prompt, "multi_modal_data": {"image": image}}
                )

            load_time = time.time() - load_start_time
            print(
                f"Batch {i//self.batch_size + 1}: Image loading time = {load_time:.2f} sec"
            )

            # Generate model prediction (output string)
            outputs = self.llm.generate(llm_inputs, sampling_params=sampling_params)

            # Process generated string (parse string, convert to dict, opt:scale )
            for img_name, output in zip(batch_names, outputs):

                output_text = output.outputs[0].text

                modified_text = clean_ocr_text(output_text)
                words, norm_boxes = parse_ocr_string(modified_text)

                cells = []
                for word, norm_box in zip(words, norm_boxes):
                    ocr_dict = {"bbox": norm_box, "text": word}
                    cells.append(ocr_dict)

                name_to_cells[img_name] = cells

            del llm_inputs, outputs

            if (i + len(batch_images)) % self.log_interval == 0 or i + len(
                batch_images
            ) == len(pil_images):
                print(f"Processed | {i + len(batch_images)} / {len(pil_images)} images")
        

        updated_dataset = dataset.map(
            self.replace_cells,
            batched=False,
            fn_kwargs={"name_to_cells": name_to_cells},
        )

        # Save locally
        print(f"Saving dataset to: {output_dir}")
        dataset_hf = DatasetDict({f"{split}": updated_dataset})
        dataset_hf.save_to_disk(output_dir)

        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f} sec")
