import logging
import random

#from constants import MAX_BBOX_SIZE
from datasets import load_from_disk
from torch.utils.data import Dataset

IGNORE_INDEX = -100
MAX_BBOX_SIZE = 500

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRDataset(Dataset):
    def __init__(self, processor, config=None, split="train"):
        """Initialize the ChemOCR dataset."""
        if config is None:
            raise ValueError("Config must not be None.")

        logger.info(
            "Building Dataset: %s for %s",
            config.get("name", "Unknown"),
            config.get("task", "Unknown"),
        )

        self._processor = processor
        self._config = config
        self._split = split

        self._load_type = config.get("load", "default")
        self._ds = load_from_disk(config.get("dataset_path"), keep_in_memory=False)[
            self._split
        ]
        self._instruction_format = config.get("instruction_format", "default")

        # Define instructions
        if self._instruction_format == "LanguagePrompt":
            self._instructions = [
                "Perform OCR on this chemical structure image.",
                "OCR this chemical structure image.",
                "Extract all text from this chemical structure image.",
                "Extract all textual content and its location in this chemical structure image.",
            ]
        else:
            self._instructions = ["OCR this chemical structure image."]

        # Preprocess dataset if required
        if config.get("preprocess", False):
            # self._ds = self._ds.map(self.create_det, batched=True)
            self._ds = self._ds.map(self.mod_create_det, batched=True)

    def create_det(self, examples):
        """Create OCR outputs by processing cell text and bounding boxes."""
        outputs = []
        for cells in examples["cells"]:
            output = ""
            for cell in cells:
                output += cell["text"]
                scaled_bbox = [int(x * MAX_BBOX_SIZE) for x in cell["bbox"]]
                scaled_bbox_string = "".join(f"<loc_{x}>" for x in scaled_bbox)
                output += scaled_bbox_string + "\n"
            outputs.append(output)
        examples["ocr"] = outputs
        return examples

    def mod_create_det(self, examples):
        """Create OCR outputs by processing cell text and bounding boxes."""
        outputs = []
        for cells in examples["cells"]:
            output = ""
            for cell in cells:
                scaled_bbox = [int(x * MAX_BBOX_SIZE) for x in cell["bbox"]]
                scaled_bbox_string = "".join(f"<loc_{x}>" for x in scaled_bbox)
                output += scaled_bbox_string
                output += cell["text"] + "\n"
            outputs.append(output)
        examples["ocr"] = outputs
        return examples

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        sample = self._ds[idx]

        image = sample["page_image"].convert("RGB")

        ocr_out = sample["ocr"]

        random_instruction = random.choice(self._instructions)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": random_instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"<ocr><loc_0><loc_0><loc_{MAX_BBOX_SIZE}><loc_{MAX_BBOX_SIZE}>{ocr_out.replace('<image>', '')}</ocr>",
                    }
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages, add_generation_prompt=False
        )

        return text, image
