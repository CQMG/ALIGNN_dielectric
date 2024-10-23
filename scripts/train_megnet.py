#!/usr/bin/env python3
 

from __future__ import annotations

import os
import shutil
import warnings
import zipfile
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import lightning as pl
import torch
from dgl.data.utils import split_dataset
from pymatgen.core import Structure
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
from matgl.layers import BondExpansion
from matgl.models import MEGNet
from matgl.utils.io import RemoteFile
from matgl.utils.training import ModelLightningModule

warnings.simplefilter("ignore")

def load_dataset() -> tuple[list[Structure], list[str], list[float]]:
    if not os.path.exists("mp.2018.6.1.json"):
        print("mp.2018.6.1.json not found, downloading... ")
        f = RemoteFile("https://figshare.com/ndownloader/files/15087992")
        with zipfile.ZipFile(f.local_path) as zf:
            zf.extractall(".")
        print("mp.2018.6.1.json downloaded.\n")
    print("Extracting mp.2018.6.1.json...")
    data = pd.read_json("mp.2018.6.1.json")
    structures = []
    mp_ids = []

    for mid, structure_str in tqdm(zip(data["material_id"], data["structure"])):
        struct = Structure.from_str(structure_str, fmt="cif")
        structures.append(struct)
        mp_ids.append(mid)

    print("Data ready.\n")
    
    return structures, mp_ids, data["formation_energy_per_atom"].tolist()

structures, mp_ids, eform_per_atom = load_dataset()

print(len(structures))
