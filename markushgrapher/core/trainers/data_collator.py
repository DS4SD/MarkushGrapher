import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def pad_sequence_native(seq, target_len, pad_value=0, dtype=torch.int):
    if isinstance(seq, torch.Tensor):
        n = seq.shape[0]
    else:
        n = len(seq)
        seq = torch.tensor(seq, dtype=dtype)
    m = target_len - n
    ret = torch.tensor([pad_value] * m, dtype=seq.dtype)
    ret = torch.cat([seq, ret], dim=0)[:target_len]
    return ret


@dataclass
class DataCollator:
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 1024
    max_length_decoder: Optional[int] = 512
    max_length_char: Optional[int] = 1024 + 512
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ):  # -> Dict[str, torch.Tensor]:
        # features: dict_keys(['input_ids', 'attention_mask', 'labels', 'seg_data', 'visual_seg_data', 'decoder_attention_mask', 'image', 'char_ids', 'char_seg_data'])
        if features[0] is None:
            return {"placeholder": torch.zeros(size=(2, 2), dtype=torch.long)}
        batch_size = len(features)
        special_labels = ["pixel_values"]
        max_len = self.max_length
        max_len_decoder = self.max_length_decoder
        max_len_char = self.max_length_char

        target_len = max_len
        target_len_decoder = max_len_decoder

        # if features[0]["char_ids"] is not None:
        if "char_ids" in features[0]:
            max_feature_len_char = max([f["char_ids"].shape[0] for f in features])
            target_len_char = min(max_feature_len_char, max_len_char)

        batch = {}
        for key in features[0].keys():

            pad_value = 0
            if key in ["bbox"]:
                pad_value = [0] * 4
            elif key in ["labels", "image_mask_labels"]:
                pad_value = -100
            elif key in special_labels:
                continue

            if key in [
                "decoder_input_ids",
                "labels",
                "decoder_attention_mask",
                "decoder_seg_data",
            ]:
                for f in features:
                    f[key] = f[key][:target_len_decoder]
                batched_feature = torch.stack(
                    [
                        pad_sequence_native(f[key], target_len_decoder, pad_value)
                        for f in features
                    ],
                    dim=0,
                )
            elif key == "visual_seg_data":
                batched_feature = torch.stack([f[key] for f in features], dim=0)
            elif key in ["char_ids", "char_seg_data"]:
                batched_feature = torch.stack(
                    [
                        pad_sequence_native(f[key], target_len_char, pad_value)
                        for f in features
                    ],
                    dim=0,
                )
            elif key == "definition_groups":
                batched_feature = torch.stack(
                    [f[key] for f in features],
                    dim=0,
                )
            elif key != "image":
                for f in features:
                    f[key] = f[key][:target_len]

                batched_feature = torch.stack(
                    [
                        pad_sequence_native(f[key], target_len, pad_value)
                        for f in features
                    ],
                    dim=0,
                )
            batch[key] = batched_feature

        if "pixel_values" in features[0]:
            image_list = torch.stack([d["pixel_values"] for d in features])
            batch.update({"pixel_values": image_list})

        return batch
