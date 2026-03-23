import argparse

from markushgrapher.ocr.chemical_ocr import Chemical_OCR


def main(dataset_dir: str, model_path: str, output_dir: str = None):
    """Run ChemicalOCR on a dataset directory.

    Args:
        dataset_dir: Path to the input HuggingFace dataset directory.
        model_path: Path to the ChemicalOCR model checkpoint.
        output_dir: Path to save OCR results. Defaults to <dataset_dir>-chemocr.
    """
    if output_dir is None:
        output_dir = f"{dataset_dir}-chemocr"

    ocr_model = Chemical_OCR(model_path=model_path)
    ocr_model.predict(dataset_dir=dataset_dir, output_dir=output_dir)


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
    args = parser.parse_args()
    main(args.dataset_dir, args.model_path, args.output_dir)
