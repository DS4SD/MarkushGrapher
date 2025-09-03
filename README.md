# MarkushGrapher

[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MarkushGrapher%0ADatasets%0A-blue)](https://huggingface.co/datasets/ds4sd/MarkushGrapher-Datasets)
[![arXiv](https://img.shields.io/badge/arXiv-2308.12234-919191.svg)](https://doi.org/10.48550/arXiv.2503.16096)
[![CVPR](https://img.shields.io/badge/Paper-CVPR52734.2025.01352-b31b1b.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Morin_MarkushGrapher_Joint_Visual_and_Textual_Recognition_of_Markush_Structures_CVPR_2025_paper.html)

This is the repository for [MarkushGrapher: Joint Visual and Textual Recognition of Markush Structures](https://arxiv.org/abs/2503.16096).

<img src="assets/architecture.png" alt="Description of the image" width="900" />

### Citation 

If you find this repository useful, please consider citing:

```
@inproceedings{Morin_2025_CVPR,
	title        = {{MarkushGrapher: Joint Visual and Textual Recognition of Markush Structures}},
	author       = {Morin, Lucas and Weber, Valery and Nassar, Ahmed and Meijer, Gerhard Ingmar and Van Gool, Luc and Li, Yawei and Staar, Peter},
	year         = 2025,
	month        = {June},
	booktitle    = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
	pages        = {14505--14515}
}
```
Links: [CVPR](https://openaccess.thecvf.com/content/CVPR2025/html/Morin_MarkushGrapher_Joint_Visual_and_Textual_Recognition_of_Markush_Structures_CVPR_2025_paper.html), [Arxiv](https://doi.org/10.48550/arXiv.2503.16096) 

### Installation

1. Create a virtual environment.
```
python3.10 -m venv markushgrapher-env
source markushgrapher-env/bin/activate
```

2. Install MarkushGrapher.
```
pip install -e .
```

3. Install [transformers](https://github.com/lucas-morin/transformers). This fork contains the code for the MarkushGrapher architecture. It was written starting from a copy of the [UDOP](https://arxiv.org/abs/2212.02623) architecture.
```
git clone https://github.com/lucas-morin/transformers.git ./external/transformers
pip install -e ./external/transformers
```

4. Install [MolScribe](https://github.com/lucas-morin/MolScribe.git). This fork contains minor fixes for compatibility with albumentations.
```
git clone https://github.com/lucas-morin/MolScribe.git ./external/MolScribe
pip install -e ./external/MolScribe --no-deps
```

### Model

Download the MarkushGrapher model from [HuggingFace](https://huggingface.co/ds4sd/MarkushGrapher/).
```
huggingface-cli download ds4sd/MarkushGrapher --local-dir ./tmp/ --repo-type model && cp -r ./tmp/models . && rm -r ./tmp/
```

Download the MolScribe model from [HuggingFace](https://huggingface.co/yujieq/MolScribe/). 
```
wget https://huggingface.co/yujieq/MolScribe/resolve/main/swin_base_char_aux_1m680k.pth -P ./external/MolScribe/ckpts/ 
```

### Datasets 

Download the datasets from [HuggingFace](https://huggingface.co/datasets/ds4sd/MarkushGrapher-Datasets).
```
huggingface-cli download ds4sd/MarkushGrapher-Datasets --local-dir ./data/hf --repo-type dataset
```

For training, we use:
1. MarkushGrapher-Synthetic-Training (Synthetic dataset)

For benchmarking, we use:
1. M2S (Multi-modal real-world dataset)
2. USPTO-Markush (Image-only real-world dataset)
3. MarkushGrapher-Synthetic (Synthetic dataset)

The synthetic datasets are generated using [MarkushGenerator](https://github.com/DS4SD/MarkushGenerator). 

### Inference

Note: MarkushGrapher is currently not able to process images without OCR annotations. The model relies on OCR bounding boxes and text provided as input. 

1. Select a dataset by setting the `dataset_path` parameter in `MarkushGrapher/config/dataset_predict.yaml`.

2. Run MarkushGrapher.
```
python3.10 -m markushgrapher.eval config/predict.yaml
```

3. Visualize predictions in: `MarkushGrapher/data/visualization/prediction/`. 

### Training

1. Select the training configuration in `MarkushGrapher/config/train.yaml` and `MarkushGrapher/config/datasets/datasets.yaml`.

2. Run training script.
```
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python3.10 -m markushgrapher.train config/train.yaml
```

### Acknowledgments

MarkushGrapher uses the code of [UDOP](https://arxiv.org/abs/2212.02623) and the [MolScribe](https://arxiv.org/abs/2205.14311) model. 

MarkushGrapher was trained from the pre-trained UDOP weights available on [HuggingFace](https://huggingface.co/ZinengTang/Udop) (checkpoint: `udop-unimodel-large-512-300k-steps.zip`).
