import argparse
import os


def check_hf_filesystem_versions():
    try:
        from packaging.version import Version
        import fsspec
        import huggingface_hub
    except Exception:
        return

    if Version(huggingface_hub.__version__) >= Version("1.0.0") and Version(
        fsspec.__version__
    ) < Version("2024.6.1"):
        raise RuntimeError(
            "Incompatible Hugging Face filesystem dependencies detected: "
            f"huggingface_hub=={huggingface_hub.__version__}, "
            f"fsspec=={fsspec.__version__}. "
            "Upgrade fsspec in this environment, for example: "
            "uv pip install --python chemicalocr-env/bin/python 'fsspec>=2024.6.1'"
        )


def load_source_dataset(dataset_name_or_path, config_name=None, split=None):
    from datasets import DatasetDict, load_dataset, load_from_disk

    if os.path.exists(dataset_name_or_path):
        dataset = load_from_disk(dataset_name_or_path)
        if isinstance(dataset, DatasetDict):
            if split is None:
                split = "test" if "test" in dataset else next(iter(dataset.keys()))
            return dataset[split]
        return dataset

    if split:
        return load_dataset(dataset_name_or_path, config_name, split=split)
    dataset = load_dataset(dataset_name_or_path, config_name)
    if isinstance(dataset, DatasetDict):
        split = "test" if "test" in dataset else next(iter(dataset.keys()))
        return dataset[split]
    return dataset


def ensure_column(dataset, column_name, default_value):
    if column_name in dataset.column_names:
        return dataset
    return dataset.add_column(column_name, [default_value for _ in range(len(dataset))])


def normalize_dataset(dataset):
    if "id" not in dataset.column_names:
        dataset = dataset.add_column("id", [str(i) for i in range(len(dataset))])

    if "page_image" not in dataset.column_names:
        if "image" in dataset.column_names:
            dataset = dataset.map(lambda row: {"page_image": row["image"]})
        else:
            raise ValueError("Dataset must contain either a 'page_image' or 'image' column.")

    dataset = ensure_column(dataset, "page_image_path", "")
    dataset = ensure_column(dataset, "description", "")
    dataset = ensure_column(dataset, "mol", "")
    dataset = ensure_column(dataset, "cxsmiles_dataset", "")
    dataset = ensure_column(dataset, "keypoints", "")
    dataset = ensure_column(dataset, "cells", [])

    if "cxsmiles" not in dataset.column_names:
        dataset = ensure_column(dataset, "cxsmiles", "")
    if "cxsmiles_opt" not in dataset.column_names:
        dataset = ensure_column(dataset, "cxsmiles_opt", "")

    def ensure_annotation(row):
        annotation = row.get("annotation")
        if annotation:
            return {"annotation": annotation}
        return {"annotation": f"<cxsmi>{row.get('cxsmiles_opt') or ''}</cxsmi>"}

    dataset = dataset.map(ensure_annotation)

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Load a local or Hub Hugging Face dataset and save one split for inference."
    )
    parser.add_argument("--dataset", required=True, help="HF dataset name or local dataset path.")
    parser.add_argument("--config", default=None, help="Optional HF dataset config name.")
    parser.add_argument("--split", default="test", help="Source split to load.")
    parser.add_argument(
        "--target_split",
        default="test",
        help="Split name to use in the saved local DatasetDict.",
    )
    parser.add_argument("--output_dir", required=True, help="Local output directory.")
    args = parser.parse_args()

    from datasets import DatasetDict

    check_hf_filesystem_versions()
    dataset = load_source_dataset(args.dataset, args.config, args.split)
    dataset = normalize_dataset(dataset)
    DatasetDict({args.target_split: dataset}).save_to_disk(args.output_dir)
    print(f"Saved {len(dataset)} samples to {args.output_dir} split '{args.target_split}'.")


if __name__ == "__main__":
    main()
