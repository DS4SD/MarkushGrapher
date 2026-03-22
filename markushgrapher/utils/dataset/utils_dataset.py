import os

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm


def split_and_save_dataset(
    dataset_path: str,
    save_path: str = "./new_dataset",
    split_name: str = "train",
    test_size: int = 100,
):
    """
    Load a Hugging Face dataset locally, split the last `test_size` samples into a test split,
    and save the result as a new local dataset with 'train' and 'test' splits.

    Args:
        dataset_path (str): Path to the local dataset directory.
        save_path (str): Path where the new dataset will be saved.
        split_name (str): Name of the original split to operate on.
        test_size (int): Number of samples to take from the end for the test split.
    """
    # Load the dataset
    dataset_path = dataset_path + f"/{split_name}"
    dataset = load_from_disk(dataset_path)

    # Check if dataset has enough samples
    if len(dataset) < test_size:
        raise ValueError(
            f"Dataset only has {len(dataset)} samples, which is less than test_size={test_size}"
        )

    # Split into train and test
    train_dataset = dataset.select(range(len(dataset) - test_size))
    test_dataset = dataset.select(range(len(dataset) - test_size, len(dataset)))

    # Create a DatasetDict
    new_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Save to disk
    os.makedirs(save_path, exist_ok=True)
    new_dataset.save_to_disk(save_path)
    print(f"New dataset saved at: {save_path}")


def save_images_and_update_paths(dataset_split: Dataset, save_dir: str) -> Dataset:
    """
    Saves the PIL images in 'page_image' field to `save_dir` as PNG files.
    Updates 'page_image_path' in the dataset with the saved file path.

    Args:
        dataset_split: A Hugging Face Dataset (e.g., molparser_dataset["train"])
        save_dir: Directory where images will be saved.

    Returns:
        A new Dataset with updated 'page_image_path' fields.
    """
    os.makedirs(save_dir, exist_ok=True)

    def save_image(example):
        img = example.get("page_image")
        if img is not None:
            file_name = (
                f"{example['id']}.png" if "id" in example else f"{example['idx']}.png"
            )
            file_path = os.path.join(save_dir, file_name)
            try:
                # Save the PIL image as PNG
                img.save(file_path, format="PNG")
                example["page_image_path"] = file_path
            except Exception as e:
                print(
                    f"Failed to save image for example {example.get('id', 'unknown')}: {e}"
                )
                example["page_image_path"] = None
        else:
            example["page_image_path"] = None
        return example

    # If dataset_split doesn't have an 'id' or 'idx' field, add an index
    if (
        "id" not in dataset_split.column_names
        and "idx" not in dataset_split.column_names
    ):
        dataset_split = dataset_split.add_column("idx", list(range(len(dataset_split))))

    updated_dataset = dataset_split.map(save_image)
    return updated_dataset


def square_with_white_borders_resize(
    image,
    output_page_width=1024,
    output_page_height=1024,
    image_scale=0.8,  # scale factor to reduce image size inside the canvas
):
    """
    Resize the image to fit within a square canvas with white borders.
    image_scale < 1.0 makes the image smaller inside the 1024x1024 canvas.
    """
    original_width, original_height = image.size

    # Apply image_scale to reduce the area the image can occupy
    target_width = int(output_page_width * image_scale)
    target_height = int(output_page_height * image_scale)

    scaling_factor = min(target_width / original_width, target_height / original_height)

    new_size = (
        int(original_width * scaling_factor),
        int(original_height * scaling_factor),
    )
    resized_image = image.resize(new_size, Image.LANCZOS)

    new_image = Image.new("RGB", (output_page_width, output_page_height), "white")
    top_left_x = (output_page_width - new_size[0]) // 2
    top_left_y = (output_page_height - new_size[1]) // 2
    new_image.paste(resized_image, (top_left_x, top_left_y))

    return new_image


def load_page_image(example, resize_image=True, image_scale=0.8):
    """
    Loads the image from 'page_image_path' and adds it to 'page_image'.
    """
    path = example.get("page_image_path")
    if path and os.path.exists(path):
        try:
            image = Image.open(path).convert("RGB")
            if resize_image:
                image = square_with_white_borders_resize(image, image_scale=image_scale)
            example["page_image"] = image
        except Exception as e:
            print(f"[Warning] Failed to load image at {path}: {e}")
            example["page_image"] = None
    else:
        example["page_image"] = None
    return example
