import argparse
import json
import os

from datasets import Dataset, DatasetDict
from markushgenerator.text_generation.image_text_merging import ImageTextMerger
from PIL import Image
from tqdm import tqdm

from markushgrapher.ocr.chemical_ocr import Chemical_OCR


def apply_ocr(hf_dataset_dir, model_path, output_dir=None):
    """Apply ChemicalOCR to all splits of a HuggingFace dataset.

    Args:
        hf_dataset_dir: Path to the HuggingFace dataset directory.
        model_path: Path to the ChemicalOCR model checkpoint.
        output_dir: Output directory. Defaults to overwriting hf_dataset_dir in-place.
    """
    import shutil
    import tempfile

    dict_path = os.path.join(hf_dataset_dir, "dataset_dict.json")
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"No dataset_dict.json found in {hf_dataset_dir}")
    with open(dict_path, "r") as f:
        data = json.load(f)
    splits = list(data["splits"])

    overwrite_in_place = output_dir is None
    if overwrite_in_place:
        output_dir = tempfile.mkdtemp(prefix="chemocr_")

    print(f"hf_dataset_dir: {hf_dataset_dir}")
    print(f"output_dir: {output_dir}")

    ocr_model = Chemical_OCR(model_path=model_path)

    for split in splits:
        input_path = f"{hf_dataset_dir}/{split}"
        ocr_model.predict(dataset_dir=input_path, output_dir=output_dir, split=split, verbose=True)

    if overwrite_in_place:
        shutil.rmtree(hf_dataset_dir)
        shutil.move(output_dir, hf_dataset_dir)
        print(f"OCR results saved back to: {hf_dataset_dir}")


def generate_hf_dataset(image_dir, output_dir, save_local=True, split="test"):

    image_text_merger = ImageTextMerger()

    samples = []
    for filename in tqdm(os.listdir(image_dir)):
        if filename.lower().endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            image_name = filename.split(".png")[0]
            cells = []  # to be filled later by Chemical OCR

            # Crop, resize, pad image
            pil_image, cells = image_text_merger.crop_resize_pad(
                image, cells, output_page_width=1024, output_page_height=1024
            )

            sample = {
                "id": image_name,
                "page_image_path": image_path,
                "description": "",
                "annotation": "",
                "mol": "",
                "cxsmiles_dataset": "",
                "cxsmiles": "",
                "cxsmiles_opt": "",
                "keypoints": "",
                "cells": cells,
                "page_image": pil_image,
            }
            samples.append(sample)

    # generate Dataset object
    dataset = Dataset.from_list(samples)

    dataset_hf = DatasetDict({f"{split}": dataset})

    if save_local:
        dataset_hf.save_to_disk(output_dir)
        print(f"Dataset saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert image directory into Hugging Face dataset"
    )

    # Get the directory of the current script
    base_dir = os.path.dirname(__file__)

    parser.add_argument(
        "--image_dir",
        type=str,
        default=os.path.join(base_dir, "../data/images/ocr_og"),
        help="Path to the directory containing input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(base_dir, "../../data/hf/sample-images"),
        help="Path where the output Hugging Face dataset will be saved",
    )
    parser.add_argument(
        "--apply_ocr",
        action="store_true",
        help="Whether to apply OCR after creating the dataset",
    )
    parser.add_argument(
        "--ocr_model_path",
        type=str,
        default=None,
        help="Path to the ChemicalOCR model checkpoint (required if --apply_ocr is set)",
    )

    args = parser.parse_args()

    generate_hf_dataset(args.image_dir, args.output_dir)

    if args.apply_ocr:
        if not args.ocr_model_path:
            parser.error("--ocr_model_path is required when --apply_ocr is set")
        apply_ocr(args.output_dir, model_path=args.ocr_model_path)


if __name__ == "__main__":
    main()
