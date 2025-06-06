from pprint import pprint

import torch
import torch.nn.functional as F
import yaml


def read_yaml_file(file_path):
    with open(file_path, "r") as file:
        yaml_content = yaml.load(file, Loader=yaml.FullLoader)
        return yaml_content


def encode_item(
    item,
    processor,
    tokenizer,
    markush_tokenizer,
    collator,
    split,
    definition_group_selector=None,
    encode_definition_group=False,
    verbose=False,
):
    """
    Note: encode_item() is called on a single sample. The batching is done afterwards.
    """
    dataset_config = item["config"]

    image, instruction, input_ids, bbox_list, labels = collator.collate(
        item, dataset_config["normalize_bbox"]
    )  # Split bounding boxes

    encoding = processor(
        images=image.convert("RGB"),
        text=[instruction],
        text_pair=[input_ids],
        boxes=[bbox_list],
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    labels_encoding = []

    for label in labels:
        if item["config"]["udop_tokenizer_only"]:
            for subtoken in tokenizer.tokenize(label):
                labels_encoding.extend(
                    tokenizer.encode(subtoken, add_special_tokens=False)
                )
        else:
            # Example of labels ["<markush><cxsmi>[R2]C1=CC=C(C(=O)NC2...-C6) alkyl group</stable></markush>", " </s>"]
            if "markush" in label:
                labels_encoding.extend(markush_tokenizer.encode_markush(label))
            elif "cxsmi" in label:
                labels_encoding.extend(markush_tokenizer.encode_cxsmi(label))
            elif "smi" in label:
                labels_encoding.extend(markush_tokenizer.encode_smi(label))
            else:
                # Encodes the </s>
                for subtoken in tokenizer.tokenize(label):
                    labels_encoding.extend(
                        tokenizer.encode(subtoken, add_special_tokens=False)
                    )

    decoder_attention_mask = torch.tensor([1] * len(labels_encoding), dtype=torch.long)

    encoding["input_ids"] = encoding["input_ids"][0]
    encoding["bbox"] = encoding["bbox"][0]
    encoding["attention_mask"] = encoding["attention_mask"][0]
    encoding["pixel_values"] = encoding["pixel_values"][0]
    encoding["labels"] = torch.tensor(labels_encoding)
    encoding["decoder_attention_mask"] = decoder_attention_mask
    if split != "train":
        encoding["image"] = image

    if encode_definition_group:
        # Create definition groups
        max_number_definition_groups = 16
        groups = definition_group_selector.select(
            encoding["input_ids"], encoding["bbox"], verbose=verbose
        )
        if verbose:
            print("len groups:", len(groups))
            print("groups:", groups)

        if groups == []:
            groups = torch.full((max_number_definition_groups, 4), -1)
        else:
            # Apply padding for batching
            groups = F.pad(
                torch.tensor(groups),
                (0, 0, 0, max_number_definition_groups - len(groups)),
                mode="constant",
                value=-1,
            )
        encoding["definition_groups"] = groups

    return encoding
