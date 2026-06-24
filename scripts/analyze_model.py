#!/usr/bin/env python3

print("Loading modules, wait for \">\" prompt...")

import os, sys, json, traceback

import torch

from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.db.jsonutils import loadjson

import plotille math
from colorama import Fore, Back, Style

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

class ModelContext:
    def __init__(self):
        self.DEVICE = torch.device("cpu")
        self.MODEL = None

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

            key = "plot_loc"
            self.PLOT_LOC = env_config[key]

            key = "cache_out"
            self.CACHE_OUT = env_config[key]

            key = "test_loc"
            self.TEST_LOC = env_config[key]

            key = "plotille_template"
            self.QUICKPLOT_TEMPLATE_PLOTILLE = env_config[key]

            key = "img_template"
            self.QUICKPLOT_TEMPLATE_IMG = env_config[key]

            key = "data_loc"
            self.REFERENCE_DATA_LOC = env_config[key]

            key = "direction"
            self.DIRECTION = env_config[key]

            key = "type"
            self.TYPE = env_config[key]
        except AttributeError:
            print(f"Malformed env_config.json: Missing key '{key}'")
            exit(1)
        
        try:
            reference_file = open(self.REFERENCE_DATA_LOC)
            self.REFERENCE_LINES = id_prop_file.readlines()
        except FileNotFoundError:
            print(f"Reference data could not be found at '{self.REFERENCE_DATA_LOC}'. {Fore.RED}Any attempt to access reference data will fail!{Fore.RESET}")
            self.REFERENCE_LINES = None

    
    def load_model(self):
        '''Loads the model from disk and prepares for querying'''
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        
        config = loadjson(self.CONFIG_LOC)
        self.MODEL = ALIGNNAtomWise(ALIGNNAtomWiseConfig(**config["model"]))

        self.MODEL.state_dict()
        self.MODEL.load_state_dict(torch.load(self.MODEL_LOC))

        self.MODEL.to(self.DEVICE)
        self.MODEL.eval()

        print("Model loaded successfully:")
        print(f"  Device:{Fore.BLUE}\t{self.DEVICE}{Fore.RESET}")
        print(f"  Type:{Fore.BLUE}\t{self.TYPE}{Fore.RESET}")
        

