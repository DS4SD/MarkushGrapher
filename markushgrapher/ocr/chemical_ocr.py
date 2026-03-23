import os
import re
import time

import torch
from datasets import DatasetDict, load_from_disk

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

def _mlx_available():
    """Check if mlx-vlm is installed without importing it."""
    import importlib.util
    return importlib.util.find_spec("mlx_vlm") is not None


def _ensure_stock_transformers():
    """Install stock transformers + tokenizers into _hf_transformers if missing."""
    import subprocess
    import sys

    hf_tf_dir = os.path.join(os.path.dirname(__file__), "_hf_transformers")
    if not os.path.exists(os.path.join(hf_tf_dir, "transformers")):
        print("Installing stock transformers for ChemicalOCR (one-time setup)...")
        os.makedirs(hf_tf_dir, exist_ok=True)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--target", hf_tf_dir, "--no-deps",
            "transformers==4.46.3", "tokenizers==0.22.2",
        ], stdout=subprocess.DEVNULL)
        # Patch dependency_versions_check to avoid version conflicts
        dep_check = os.path.join(hf_tf_dir, "transformers", "dependency_versions_check.py")
        if os.path.exists(dep_check):
            with open(dep_check, "w") as f:
                f.write("def dep_version_check(pkg, hint=None):\n    pass\n")
        # Patch idefics3 image processor: resize() must pass max_image_size
        img_proc = os.path.join(hf_tf_dir, "transformers", "models", "idefics3", "image_processing_idefics3.py")
        if os.path.exists(img_proc):
            with open(img_proc) as f:
                content = f.read()
            old = '            size = get_resize_output_image_size(\n                image, resolution_max_side=size["longest_edge"], input_data_format=input_data_format\n            )'
            new = '            max_img_size = self.max_image_size.get("longest_edge", 1820) if hasattr(self, "max_image_size") and isinstance(self.max_image_size, dict) else 1820\n            size = get_resize_output_image_size(\n                image, resolution_max_side=size["longest_edge"], max_image_size=max(max_img_size, size["longest_edge"]), input_data_format=input_data_format\n            )'
            if old in content:
                with open(img_proc, "w") as f:
                    f.write(content.replace(old, new))
        print("Stock transformers installed successfully.")
    return hf_tf_dir


def _load_stock_transformers_model(model_path, device):
    """Load processor and model using stock transformers (not the custom fork).

    The MarkushGrapher repo uses a custom transformers fork (v4.34) that does not
    include Idefics3. This function temporarily swaps in stock transformers (4.46.3)
    from markushgrapher/ocr/_hf_transformers/ to load the model, then restores
    the original modules.
    """
    import sys

    hf_tf_dir = _ensure_stock_transformers()
    sys.path.insert(0, hf_tf_dir)

    # Remove cached transformers/tokenizers modules so Python picks up the stock version
    cached = {
        k: v for k, v in sys.modules.items()
        if k == "transformers" or k.startswith("transformers.")
        or k == "tokenizers" or k.startswith("tokenizers.")
    }
    for k in cached:
        del sys.modules[k]

    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq

        # Load processor and model while stock transformers is active
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        ).to(device)
        model.eval()
    finally:
        # Restore the original transformers modules and remove our path
        if hf_tf_dir in sys.path:
            sys.path.remove(hf_tf_dir)
        for k in list(sys.modules.keys()):
            if (k == "transformers" or k.startswith("transformers.")
                    or k == "tokenizers" or k.startswith("tokenizers.")):
                del sys.modules[k]
        sys.modules.update(cached)

    return processor, model


def _convert_to_mlx(model_path, mlx_path):
    """Convert a HuggingFace model to MLX format using mlx_vlm.convert."""
    import subprocess
    import sys

    hf_tf_dir = _ensure_stock_transformers()
    env = os.environ.copy()
    env["PYTHONPATH"] = hf_tf_dir + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call([
        sys.executable, "-m", "mlx_vlm.convert",
        "--hf-path", model_path,
        "--mlx-path", mlx_path,
    ], env=env)


def _load_mlx_model(mlx_path):
    """Load MLX model with stock transformers for processor compatibility.

    mlx_vlm internally uses AutoProcessor which needs stock transformers
    (not the custom fork) to recognize Idefics3Processor. Both mlx_vlm and
    stock transformers are imported fresh inside this function.
    """
    import sys

    hf_tf_dir = _ensure_stock_transformers()
    sys.path.insert(0, hf_tf_dir)

    # Save and clear transformers/tokenizers modules (keep mlx_vlm out so it
    # gets imported fresh against stock transformers)
    cached_tf = {
        k: v for k, v in sys.modules.items()
        if k == "transformers" or k.startswith("transformers.")
        or k == "tokenizers" or k.startswith("tokenizers.")
    }
    cached_mlx = {
        k: v for k, v in sys.modules.items()
        if k == "mlx_vlm" or k.startswith("mlx_vlm.")
    }
    for k in list(cached_tf) + list(cached_mlx):
        del sys.modules[k]

    try:
        import mlx_vlm as _mlx_vlm
        model, processor = _mlx_vlm.load(mlx_path)
        config = _mlx_vlm.utils.load_config(mlx_path)
    finally:
        # Restore fork transformers, but keep mlx_vlm modules loaded
        if hf_tf_dir in sys.path:
            sys.path.remove(hf_tf_dir)
        for k in list(sys.modules.keys()):
            if (k == "transformers" or k.startswith("transformers.")
                    or k == "tokenizers" or k.startswith("tokenizers.")):
                del sys.modules[k]
        sys.modules.update(cached_tf)

    return model, processor, config


def _get_device():
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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

        Uses vllm for GPU inference when available, falls back to
        transformers for CPU/MPS inference.

        Args:
            model_path (str): Path to model checkpoint.
            batch_size (int): Batch size for inference.
            log_interval (int): Logging interval.
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.log_interval = log_interval

        # Backend priority: vllm (CUDA) > mlx (Apple Silicon) > transformers (CPU)
        if LLM is not None and torch.cuda.is_available():
            self.backend = "vllm"
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.llm = LLM(model=self.model_path, limit_mm_per_prompt={"image": 1})
            print(
                f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
            print(
                f"Allocated GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
            )
            print(f"Cached GPU Memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        elif _mlx_available() and torch.backends.mps.is_available():
            self.backend = "mlx"
            mlx_path = model_path.rstrip("/") + "-mlx"
            if not os.path.exists(mlx_path):
                print(f"Converting model to MLX format: {mlx_path}")
                _convert_to_mlx(model_path, mlx_path)
            self.mlx_model, self.mlx_processor, self.mlx_config = _load_mlx_model(mlx_path)
            print(f"ChemicalOCR loaded with MLX backend (Apple Silicon)")
        else:
            self.backend = "transformers"
            self.device = _get_device()
            self.processor, self.model = _load_stock_transformers_model(
                model_path, self.device
            )
            print(f"ChemicalOCR loaded with transformers backend on {self.device}")

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
        if self.backend == "mlx":
            from mlx_vlm.prompt_utils import apply_chat_template
            return apply_chat_template(
                self.mlx_processor, self.mlx_config, prompt, num_images=1
            )

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

    def _generate_vllm(self, batch_images, prompt):
        """Generate predictions using vllm backend."""
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096,
        )
        llm_inputs = [
            {"prompt": prompt, "multi_modal_data": {"image": image}}
            for image in batch_images
        ]
        outputs = self.llm.generate(llm_inputs, sampling_params=sampling_params)
        texts = [output.outputs[0].text for output in outputs]
        del llm_inputs, outputs
        return texts

    def _generate_mlx(self, batch_images, prompt):
        """Generate predictions using mlx-vlm backend (Apple Silicon)."""
        import os
        import tempfile
        texts = []
        for image in batch_images:
            # mlx-vlm expects image paths, so save PIL image to a temp file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                image.save(f, format="PNG")
                tmp_path = f.name
            try:
                from mlx_vlm import generate as mlx_generate
                result = mlx_generate(
                    self.mlx_model, self.mlx_processor, prompt,
                    image=[tmp_path],
                    max_tokens=4096,
                    temperature=0.0,
                    verbose=False,
                )
                texts.append(result.text)
            finally:
                os.unlink(tmp_path)
        return texts

    def _generate_transformers(self, batch_images, prompt):
        """Generate predictions using transformers backend (CPU/MPS)."""
        texts = []
        for image in batch_images:
            inputs = self.processor(
                text=prompt,
                images=[image],
                return_tensors="pt",
                size={"longest_edge": 512},
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                )

            # Decode only the generated tokens (skip the prompt)
            prompt_len = inputs["input_ids"].shape[-1]
            output_text = self.processor.batch_decode(
                generated_ids[:, prompt_len:],
                skip_special_tokens=True,
            )[0]
            texts.append(output_text)
        return texts

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

        start_time = time.time()
        name_to_cells = {}

        for i in range(0, len(pil_images), self.batch_size):

            # Load image and image name
            batch_images = pil_images[i : i + self.batch_size]
            batch_names = image_names[i : i + self.batch_size]

            load_start_time = time.time()
            load_time = time.time() - load_start_time
            print(
                f"Batch {i//self.batch_size + 1}: Image loading time = {load_time:.2f} sec"
            )

            # Generate predictions using the appropriate backend
            if self.backend == "vllm":
                output_texts = self._generate_vllm(batch_images, prompt)
            elif self.backend == "mlx":
                output_texts = self._generate_mlx(batch_images, prompt)
            else:
                output_texts = self._generate_transformers(batch_images, prompt)

            # Process generated strings
            for img_name, output_text in zip(batch_names, output_texts):
                modified_text = clean_ocr_text(output_text)
                words, norm_boxes = parse_ocr_string(modified_text)

                cells = []
                for word, norm_box in zip(words, norm_boxes):
                    ocr_dict = {"bbox": norm_box, "text": word}
                    cells.append(ocr_dict)

                name_to_cells[img_name] = cells

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
