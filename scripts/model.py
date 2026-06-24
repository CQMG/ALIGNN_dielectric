#!/usr/bin/env python3

import torch, os, json

from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.db.jsonutils import loadjson

class FunctionModel:
    def __init__(self):
        self.DEVICE = torch.device("cpu")
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")

        try:
            f = open('./env_config.json')
            env_config = json.load(f)
        except FileNotFoundError:
            print("Could not find env_config.json. There are two possibilities:\n - You didn't use the templates to create this model.\n - This model was built using an older version of these scripts, in which case\n    you should use the legacy version of this script.")
            exit(1)
        
        key = "__NO KEY__"
        
        try:
            key = "model_loc"
            self.MODEL_LOC = env_config[key]
            self.CONFIG_LOC = os.path.join(self.MODEL_LOC, "config.json")

            key = "model_file"
            self.MODEL_FILE = os.path.join(self.MODEL_LOC, env_config[key])

            key = "direction"
            self.DIRECTION = env_config[key]

            key = "type"
            self.TYPE = env_config[key]
        except KeyError:
            print(f"Malformed env_config.json: Missing key '{key}'")
            exit(1)
        
        config = loadjson(self.CONFIG_LOC)
        self.MODEL = ALIGNNAtomWise(ALIGNNAtomWiseConfig(**config["model"]))

        self.MODEL.state_dict()
        self.MODEL.load_state_dict(torch.load(self.MODEL_FILE))

        self.MODEL.to(self.DEVICE)
        self.MODEL.eval()
        
        print("Model loaded successfully:")
        print(f"  Device:\t{self.DEVICE}")
        print(f"  Type:  \t{self.TYPE}")

    
    def query(self, atoms):
        cvn = Spacegroup3D(atoms).conventional_standard_structure

        g, lg = Graph.atom_dgl_multigraph(atoms)
        return list((self.MODEL([g.to(self.DEVICE), lg.to(self.DEVICE)]))["out"].detach().cpu().numpy().flatten().tolist())
    
    def query_list(self, atoms_list):
        out_list = []
        for atoms in atoms_list:
            out_list.append(self.query(atoms))
        return out_list