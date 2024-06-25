#!/usr/bin/env python3

print("Loading, wait for \">\" prompt...")

import traceback, json, sys

import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.analysis.solarefficiency.solar import SolarEfficiency
from jarvis.db.jsonutils import loadjson

from scipy.constants import physical_constants
from scipy.constants import speed_of_light

import plotille, math
from colorama import Fore, Back, Style

import numpy as np

import matplotlib.pyplot as plt

class CommandParser:
    def __init__(self):
        
        self.REAL_DIR = "E500SmaxREAL"
        self.IMAG_DIR = "E500SmaxIMAG_nospike"



        self.MODEL_LOC_REAL = self.REAL_DIR + "/output_data/best_model.pt"
        self.MODEL_LOC_IMAG = self.IMAG_DIR + "/output_data/best_model.pt"

        self.CONFIG_LOC_REAL = self.REAL_DIR + "/output_data/config.json"
        self.CONFIG_LOC_IMAG = self.IMAG_DIR + "/output_data/config.json"

        self.NORM_DIV_LOC_REAL = self.REAL_DIR + "/output_data/norm_div.txt"
        self.NORM_DIV_LOC_IMAG = self.IMAG_DIR + "/output_data/norm_div.txt"

        self.DEVICE = torch.device("cpu")

        self.MODEL_REAL = None
        self.MODEL_IMAG = None

        self.STRUCTURES_LOC = "E500SmaxREAL/structure_data/"

    def load_models(self):
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")

        config_real = loadjson(self.CONFIG_LOC_REAL)
        config_imag = loadjson(self.CONFIG_LOC_IMAG)
        
        self.MODEL_REAL = ALIGNNAtomWise(ALIGNNAtomWiseConfig(**config_real["model"]))
        self.MODEL_IMAG = ALIGNNAtomWise(ALIGNNAtomWiseConfig(**config_imag["model"]))

        self.MODEL_REAL.state_dict()
        self.MODEL_REAL.load_state_dict(torch.load(self.MODEL_LOC_REAL))

        self.MODEL_IMAG.state_dict()
        self.MODEL_IMAG.load_state_dict(torch.load(self.MODEL_LOC_IMAG))

        self.MODEL_IMAG.to(self.DEVICE)
        self.MODEL_REAL.to(self.DEVICE)

        self.MODEL_IMAG.eval()
        self.MODEL_REAL.eval()

        print("Models loaded successfully:")
        print("  Device:\t" + str(self.DEVICE))
    

    def slme(self, ID, dirgap, indirgap):
        real = np.array(self.query(ID, "REAL"))
        imag = np.array(self.query(ID, "IMAG"))

        # Modified from:
        # https://github.com/usnistgov/jarvis/blob/52eb756d1a5512779502bb6cec564af2fd322c6a/jarvis/io/vasp/outputs.py#L1499-L1501
        eV_to_recip_cm = 1.0 / (
            physical_constants["Planck constant in eV s"][0]
            * speed_of_light
            * 1e2
        )
        
        energies = []
        curr = 0
        for i in range(0, 300):
            energies.append(curr)
            curr += 0.05
        energies = np.array(energies)
        epsilon_1 = real
        epsilon_2 = imag
        absorption = (
            2
            * np.pi
            * np.sqrt(2.0)
            * eV_to_recip_cm
            * energies
            * np.sqrt(-epsilon_1 + np.sqrt(epsilon_1**2 + epsilon_2**2))
        )
        # -----

        absorption = absorption * 100 # Not sure why this happens, see (perhaps the input needs to be a percentage?):
        # https://github.com/usnistgov/jarvis/blob/52eb756d1a5512779502bb6cec564af2fd322c6a/jarvis/db/vasp_to_xml.py#L810        
    
        seff = SolarEfficiency().slme(energies, absorption, dirgap, indirgap)

        print(seff)


    def query(self, ID, model_type):
        match model_type:
            case "REAL":
                model = self.MODEL_REAL
                norm_div_loc = self.NORM_DIV_LOC_REAL
            case "IMAG":
                model = self.MODEL_IMAG
                norm_div_loc = self.NORM_DIV_LOC_IMAG
            case _:
                print("Must select either 'REAL' or 'IMAG'")
                return None
        
        if model == None:
            print("Must call load_models first")
            return None
        
        try:
            atoms = Atoms.from_poscar(ID)
        except FileNotFoundError:
            try:
                atoms = Atoms.from_poscar(self.STRUCTURES_LOC + ID)
            except FileNotFoundError:
                print("Requested file not found")
                return None
        
        cvn = Spacegroup3D(atoms).conventional_standard_structure

        try:
            f = open(norm_div_loc, 'r')
            norm_div = float(f.readlines()[0])
            f.close()
        except Exception:
            norm_div = 1.0
        
        g, lg = Graph.atom_dgl_multigraph(atoms)

        out = list(map(lambda x: x * norm_div, (model([g.to(self.DEVICE), lg.to(self.DEVICE)]))["out"].detach().cpu().numpy().flatten().tolist()))
        
        return out



    def plotille_plot(self, ID, model_type):
        Y = self.query(ID, model_type)

        fig = plotille.Figure()
        fig.width = 120
        fig.height = 35
        fig.set_x_limits(min_=0, max_=15)

        X = np.arange(0, 15, 0.05)

        yMax = max(0, max(Y))
        yMin = min(0, min(Y))

        fig.set_y_limits(min_=yMin, max_=math.ceil(yMax))

        fig.plot(X, Y, lc="blue", label=str(ID))

        print(fig.show(legend=True))



    def parse(self, statement):
        cmpts = statement.split(" ")

        match cmpts[0]:
            case "quit":
                exit()
            case "help":
                self.help()
            case "load_models":
                self.load_models()
            case "plot":
                if len(cmpts) != 3:
                    print("Incorrect arguments, see \"help\"")
                else:
                    self.plotille_plot(cmpts[2], cmpts[1])
            case "slme":
                if len(cmpts) != 4:
                    print("Incorrect arguments, see \"help\"")
                else:
                    self.slme(cmpts[1], float(cmpts[2]), float(cmpts[3]))
            case _:
                print("Command not found, use \"help\"")

    def help(self):
        print(" <-- Help --> ")
        print(" Command List ({} are optional arguments):")
        print(Fore.RED + "  - help" + Fore.RESET)
        print("     Show this help info\n")
        print(Fore.RED + "  - quit" + Fore.RESET)
        print("     Quit the program\n")
        print(Fore.RED + "  - load_models" + Fore.RESET)
        print("     Load the checkpoints\n")
        print(Fore.RED + "  - plot [model_type] [ID]" + Fore.RESET)
        print("     Plot quickly with Plotille on terminal. model_type is either 'REAL' or 'IMAG'")
        print("     ID is the name of the structure file to evaluate for\n")
        print(Fore.RED + "  - slme [ID] [direct bandgap] [indirect bandgap]" + Fore.RESET)
        print("     Plot quickly with Plotille on terminal. model_type is either 'REAL' or 'IMAG'")
        print("     ID is the name of the structure file to evaluate for\n")

    def start(self):
        while True:
            try:
                statement = input("\n> ")
                self.parse(statement)
            except Exception as e:
                traceback.print_exc()
    
    def parse_file(self, file):
        statements = []
        try:
            f = open(file, 'r')
            statements = f.readlines()
            f.close()
        except Exception:
            print("Could not open file")
            traceback.print_exc()
            exit(2)
        
        for i in range(0, len(statements)):
            try:
                if statements[i] != "":
                    self.parse(statements[i].strip())
            except Exception:
                traceback.print_exc()
                exit(2)



if __name__ == "__main__":
    parser = CommandParser()
    if len(sys.argv) == 1:
        parser.start()
    else:
        parser.parse_file(sys.argv[1])