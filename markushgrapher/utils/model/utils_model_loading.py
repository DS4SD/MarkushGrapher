import os
import torch
import json
import numpy as np

def save_weights_separately(
    model,
    architecture_variant,
    vtl_encoder_save_dir,
    ocsr_encoder_save_dir,
    mlp_projector_save_dir,
    decoder_save_dir,
    lm_head_save_dir,
):

    os.makedirs(ocsr_encoder_save_dir, exist_ok=True)
    os.makedirs(decoder_save_dir, exist_ok=True)
    os.makedirs(mlp_projector_save_dir, exist_ok=True)

    if architecture_variant == "me-lf-stack-1-molscribe-only":
        # Save ocsr encoder weights
        torch.save(
            model.encoder.molscribe_encoder.state_dict(), #  model.encoder.state_dict()
            #model.encoder.state_dict(),
            os.path.join(ocsr_encoder_save_dir, "ocsr_encoder_weights.pth"),
        )

        # Save decoder weights
        torch.save(
            model.decoder.state_dict(),
            os.path.join(decoder_save_dir, "decoder_weights.pth"),
        )

        # Save projector weights (assuming projector is patch_embed here)
        torch.save(
            model.encoder.molscribe_projector.state_dict(),
            os.path.join(mlp_projector_save_dir, "projector_weights.pth"),
        )

        # Optionally save lm_head or shared embeddings if needed
        torch.save(model.lm_head.state_dict(), os.path.join(lm_head_save_dir, "lm_head_weights.pth"))
        #torch.save(model.shared.state_dict(), os.path.join(lm_head_save_dir, "shared_embeddings.pth"))

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False

def compute_weight_stats(module_name, state_dict, weights_filepath, verbose=True, save_to_json=False):

    # Flatten all tensor values
    all_weights = torch.cat([v.flatten() for v in state_dict.values() if isinstance(v, torch.Tensor)])

    # Compute sum of 
    first_1000_sum = all_weights[:1000].sum().item()
    last_1000_sum = all_weights[-1000:].sum().item()

    state_dict_items = {}
    weights_sum_list = []

    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            flattened = tensor.flatten().tolist()
            state_dict_items[name] = list(tensor.shape)
            weights_sum_list.append(float(np.sum(flattened)))
    
    stats_dict = {
        "num_parameters": all_weights.numel(),
        "first_1000_sum": first_1000_sum,
        "last_1000_sum": last_1000_sum,
        "state_dict_items": state_dict_items,
        "weights_sum_list": weights_sum_list
    }

    if verbose:
        print(f"----------- State: {module_name} ------------")
        print(f"[{module_name}] Number of parameters loaded: {all_weights.numel()}")
        print(f"[{module_name}] Sum of first 1000 weights: {first_1000_sum:.6f}")
        print(f"[{module_name}] Sum of last 1000 weights: {last_1000_sum:.6f}")
        print(f"[{module_name}] weights_sum_list: {weights_sum_list}")
        print("------------------------------------------------")
    
    if save_to_json:

        mlp_projector_weights_dir = os.path.dirname(weights_filepath)
        submodule = os.path.splitext(os.path.basename(weights_filepath))[0]
        stats_save_path = os.path.join(mlp_projector_weights_dir, f"weight_stats/{submodule}.json")

        with open(stats_save_path, "w") as f:
            json.dump(stats_dict, f, indent=2)
        print(f"[{submodule}] stats saved to {stats_save_path}")


def compare_module_weights(state_dict, weight_stats_path, tol=1e-9, verbose=True):
    """
    Compare two JSON files containing dictionaries for equality.
    
    Args:
        file1 (str): Path to the first JSON file.
        file2 (str): Path to the second JSON file.
        tol (float): Tolerance for floating-point comparison.
        
    Returns:
        bool: True if the dictionaries are equal, False otherwise.
    """
    def compare_values(v1, v2):
        """Recursively compare two values with tolerance for floats."""
        if isinstance(v1, dict) and isinstance(v2, dict):
            return compare_dicts(v1, v2)
        elif isinstance(v1, list) and isinstance(v2, list):
            if len(v1) != len(v2):
                return False
            return all(compare_values(a, b) for a, b in zip(v1, v2))
        elif isinstance(v1, float) or isinstance(v2, float):
            return abs(v1 - v2) < tol
        else:
            return v1 == v2

    def compare_dicts(d1, d2):
        """Recursively compare two dictionaries."""
        if d1.keys() != d2.keys():
            print("keys are not the same")
            return False
        return all(compare_values(d1[k], d2[k]) for k in d1)
    
    # Flatten all tensor values
    all_weights = torch.cat([v.flatten() for v in state_dict.values() if isinstance(v, torch.Tensor)])

    # Compute sum of 
    first_1000_sum = all_weights[:1000].sum().item()
    last_1000_sum = all_weights[-1000:].sum().item()

    state_dict_items = {}
    weights_sum_list = []

    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            flattened = tensor.flatten().tolist()
            state_dict_items[name] = list(tensor.shape)
            weights_sum_list.append(float(np.sum(flattened)))
    
    stats_dict = {
        "num_parameters": all_weights.numel(),
        "first_1000_sum": first_1000_sum,
        "last_1000_sum": last_1000_sum,
        "state_dict_items": state_dict_items,
        "weights_sum_list": weights_sum_list
    }

    if verbose:
        print(f"----------- State: loaded model weights ------------")
        print(f"[loaded model weights] Number of parameters loaded: {all_weights.numel()}")
        print(f"[loaded model weights] Sum of first 1000 weights: {first_1000_sum:.6f}")
        print(f"[loaded model weights] Sum of last 1000 weights: {last_1000_sum:.6f}")
        print(f"[loaded model weights] weights_sum_list: {weights_sum_list}")
        print("------------------------------------------------")
    

    # Load the JSON file
    with open(weight_stats_path, 'r') as f:
        stats_dict_2 = json.load(f)
    
    if verbose:

        num_parameters_stats_dict_2 = stats_dict_2["num_parameters"]
        first_1000_sum_stats_dict_2 = stats_dict_2["first_1000_sum"]
        last_1000_sum_stats_dict_2 = stats_dict_2["last_1000_sum"]
        #stats_dict_2["state_dict_items"]
        weights_sum_list_stats_dict_2 = stats_dict_2["weights_sum_list"]

        print(f"----------- State: json file ------------")
        print(f"[json file] Number of parameters loaded: {num_parameters_stats_dict_2}")
        print(f"[json file] Sum of first 1000 weights: {first_1000_sum_stats_dict_2:.6f}")
        print(f"[json file] Sum of last 1000 weights: {last_1000_sum_stats_dict_2:.6f}")
        print(f"[json file] weights_sum_list: {weights_sum_list_stats_dict_2}")
        print("------------------------------------------------")

    return compare_dicts(stats_dict, stats_dict_2)
