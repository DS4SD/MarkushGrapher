#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle

from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
from datasets import load_from_disk

from markushgrapher.utils.ocsr.utils_markush import canonicalize_markush


def get_training_smiles(
    dataset_path,
    cxsmiles_tokenizer,
    read_training_smiles=True,
    overwrite_training_smiles=False,
    verbose=False,
):
    print("Calling get_training_smiles")
    training_smiles = []
    dataset_dict = load_from_disk(dataset_path, keep_in_memory=False)
    if "train" in dataset_dict:
        read = False
        if read_training_smiles and os.path.exists(
            f"./data/training_smiles/{dataset_path.split('/')[-1]}_training_smiles.pkl"
        ):
            with open(
                f"./data/training_smiles/{dataset_path.split('/')[-1]}_training_smiles.pkl",
                "rb",
            ) as handle:
                training_smiles = pickle.load(handle)
            read = True
        if not (read):
            ds_train = dataset_dict["train"]
            if "smiles" in ds_train.column_names:
                training_smiles = list(ds_train["smiles"])
                # Canonicalize SMILES
                training_smiles = [
                    Chem.MolToSmiles(Chem.MolFromSmiles(s))
                    for s in tqdm(training_smiles)
                ]
            elif "cxsmiles_opt" in ds_train.column_names:
                training_smiles = []
                for s in tqdm(list(ds_train["cxsmiles_opt"])):
                    cxsmiles_out = cxsmiles_tokenizer.convert_opt_to_out(s)
                    if cxsmiles_out is None:
                        continue
                    # Canonicalize CXSMILES
                    canon_cxsmiles_out = canonicalize_markush(cxsmiles_out)
                    training_smiles.append(canon_cxsmiles_out)
            if overwrite_training_smiles:
                with open(
                    f"./data/training_smiles/{dataset_path.split('/')[-1]}_training_smiles.pkl",
                    "wb",
                ) as handle:
                    pickle.dump(
                        training_smiles, handle, protocol=pickle.HIGHEST_PROTOCOL
                    )
    if verbose:
        print("Training SMILES:", training_smiles)
        for s in training_smiles:
            print(s)
    print("Done get_training_smiles")
    return training_smiles
