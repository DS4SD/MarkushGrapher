import argparse
import json
import os


def get_dataset_splits(dataset_dir: str, split: str = None):
    dict_path = os.path.join(dataset_dir, "dataset_dict.json")
    if not os.path.exists(dict_path):
        return [(dataset_dir, split or "train")]

    with open(dict_path, "r") as f:
        data = json.load(f)

    splits = [split] if split else list(data["splits"])
    return [(os.path.join(dataset_dir, split_name), split_name) for split_name in splits]


def main(dataset_dir: str, model_path: str, output_dir: str = None, split: str = None):
    """Run ChemicalOCR on a dataset directory.

    Args:
        dataset_dir: Path to the input HuggingFace dataset directory.
        model_path: Path to the ChemicalOCR model checkpoint.
        output_dir: Path to save OCR results. Defaults to <dataset_dir>-chemocr.
        split: Optional split name when dataset_dir is a DatasetDict.
    """
    from markushgrapher.ocr.chemical_ocr import Chemical_OCR

    if output_dir is None:
        output_dir = f"{dataset_dir}-chemocr"

    ocr_model = Chemical_OCR(model_path=model_path)
    for split_dataset_dir, split_name in get_dataset_splits(dataset_dir, split):
        ocr_model.predict(
            dataset_dir=split_dataset_dir,
            output_dir=output_dir,
            split=split_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Chemical OCR on a specified dataset directory."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the input HuggingFace dataset directory.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the ChemicalOCR model checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to save OCR results. Defaults to <dataset_dir>-chemocr.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split to process when dataset_dir is a DatasetDict.",
    )
    args = parser.parse_args()
    main(args.dataset_dir, args.model_path, args.output_dir, args.split)
