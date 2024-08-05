#!/usr/bin/env python3

print("Loading, wait for \">\" prompt...")

import traceback, json, sys, os

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
        
        #self.REAL_DIR = os.path.join("./", input("Real model directory: "))
        self.REAL_DIR = os.path.join(".", "REAL")
        #self.IMAG_DIR = os.path.join("./", input("Imaginary model directory: "))
        self.IMAG_DIR = os.path.join(".", "IMAG")

        self.DEFAULT_MODEL_LOC = True


        self.STRUCTURES_LOC = os.path.join(".", "DATA")

        self.DEFAULT_STRUCT_LOC = True

        self.METADATA_LOC = os.path.join(self.STRUCTURES_LOC, "metadata.json")

        self.METADATA = None

        try:
            with open(self.METADATA_LOC, 'r') as f:
                self.METADATA = json.load(f)
        except FileNotFoundError:
            print("Unable to load metadata in default location (./DATA/metadata.json),")
            print("Force a reload of metadata with 'set_data'")



        self.MODEL_LOC_REAL = os.path.join(self.REAL_DIR, "output_data/best_model.pt")
        self.MODEL_LOC_IMAG = os.path.join(self.IMAG_DIR, "output_data/best_model.pt")

        self.CONFIG_LOC_REAL = os.path.join(self.REAL_DIR, "output_data/config.json")
        self.CONFIG_LOC_IMAG = os.path.join(self.IMAG_DIR, "output_data/config.json")

        self.DEVICE = torch.device("cpu")

        self.MODEL_REAL = None
        self.MODEL_IMAG = None
    
    def set_models(self, real_dir, imag_dir):
        self.REAL_DIR = os.path.join(".", real_dir)
        self.IMAG_DIR = os.path.join(".", imag_dir)

        self.MODEL_LOC_REAL = os.path.join(self.REAL_DIR, "output_data/best_model.pt")
        self.MODEL_LOC_IMAG = os.path.join(self.IMAG_DIR, "output_data/best_model.pt")

        self.CONFIG_LOC_REAL = os.path.join(self.REAL_DIR, "output_data/config.json")
        self.CONFIG_LOC_IMAG = os.path.join(self.IMAG_DIR, "output_data/config.json")

        self.DEFAULT_MODEL_LOC = False
        self.DEFAULT_STRUCT_LOC = False

        self.MODEL_REAL = None
        self.MODEL_IMAG = None

        print("Models switched, use load_models before attempting to query them\n")
    
    def set_data(self, new_dir):
        self.STRUCTURES_LOC = os.path.join(".", new_dir)

        self.DEFAULT_STRUCT_LOC = False

        self.METADATA_LOC = os.path.join(self.STRUCTURES_LOC, "metadata.json")

        try:
            with open(self.METADATA_LOC, 'r') as f:
                self.METADATA = json.load(f)
        except FileNotFoundError:
            print("Unable to load metadata in specified location (" + str(self.METADATA_LOC) + "),")
            print("Force a reload of metadata with 'set_data'")



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

        if self.DEFAULT_MODEL_LOC:
            print(Fore.RED + "WARNING: " + Fore.RESET + "Using the default models in './REAL' and './IMAG'")
            print("         This is not recommended, switch them with 'set_models'\n")
        
        if self.DEFAULT_STRUCT_LOC:
            print(Fore.RED + "WARNING: " + Fore.RESET + "Using the default structure data in './DATA'")
            print("         This is not recommended, switch it with 'set_data'\n")

        print("Models loaded successfully:")
        print("  Device:\t" + str(self.DEVICE))

    
    def automated_test(self, test_file, output_file):
        f = open(test_file)
        test_structs = list(map(lambda x: x.strip(), f.readlines()))
        f.close()

        valid_structs = []
        seffs = []
        ref_seffs = []

        errors = []

        for struct in test_structs:
            seff = self.slme(struct, self.METADATA[struct]["mbj_dir_gap"], self.METADATA[struct]["mbj_indir_gap"], True)
            if not math.isinf(seff):
                ref_seffs.append(self.METADATA[struct]["ref_slme"])
                seffs.append(seff)
                valid_structs.append(struct)

                errors.append(abs(seffs[-1] - ref_seffs[-1]))
        
        mae = MAE(seffs, ref_seffs)

        print("MAE: " + str(mae))

        entries = {}

        for i in range(0, len(valid_structs)):
            entries[valid_structs[i]] = {}

            entries[valid_structs[i]]["predicted_slme"] = seffs[i]
            entries[valid_structs[i]]["error"] = errors[i]

        try:
            f = open(output_file, "w")
            json.dump(entries, f, indent=4)
            f.close()
            print("Successfully saved to " + output_file)
        except Exception:
            print("Could not write to JSON file")
            raise
    

    def slme(self, ID, dirgap=None, indirgap=None, silent=False):
        real = np.array(self.query(ID, "REAL"))
        imag = np.array(self.query(ID, "IMAG"))

        if dirgap == None:
            if self.METADATA == None:
                print("Missing metadata, either provide valid metadata or specify bandgaps directly")
                return None
            dirgap = self.METADATA[ID]["mbj_dir_gap"]

        if indirgap == None:
            if self.METADATA == None:
                print("Missing metadata, either provide valid metadata or specify bandgaps directly")
                return None
            indirgap = self.METADATA[ID]["mbj_indir_gap"]


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
    
        if not silent:
            print("\nUsing the following parameters: ")
            print(" Struct:       " + ID)
            print(" Direct Gap:   " + str(dirgap))
            print(" Indirect Gap: " + str(indirgap))

        seff = SolarEfficiency().slme(energies, absorption, dirgap, indirgap) * 100

        
        if silent:
            print(ID)
            return seff
        

        print("\nSLME: " + str(seff) + "%")
        print()

        if self.METADATA == None:
            print(Fore.RED + "Metadata missing" + Fore.RESET)
        else:
            print("Reference info: ")
            print(" Direct Gap:   " + str(self.METADATA[ID]["mbj_dir_gap"]))
            print(" Indirect Gap: " + str(self.METADATA[ID]["mbj_indir_gap"]))
            print(" SLME:         " + str(self.METADATA[ID]["ref_slme"]) + "%")
            print(" SQ:           " + str(self.METADATA[ID]["ref_sq"]) + "%")

        
        return seff



    def query(self, ID, model_type):
        match model_type:
            case "REAL":
                model = self.MODEL_REAL
            case "IMAG":
                model = self.MODEL_IMAG
            case _:
                print(Fore.RED + "Must select either 'REAL' or 'IMAG'" + Fore.RESET)
                return None
        
        if model == None:
            print(Fore.RED + "Must call load_models first" + Fore.RESET)
            return None
        
        try:
            atoms = Atoms.from_poscar(ID)
        except FileNotFoundError:
            try:
                atoms = Atoms.from_poscar(os.path.join(self.STRUCTURES_LOC, ID))
            except FileNotFoundError:
                print(Fore.RED + "Requested file not found" + Fore.RESET)
                return None
        
        cvn = Spacegroup3D(atoms).conventional_standard_structure
        
        g, lg = Graph.atom_dgl_multigraph(atoms)

        out = list((model([g.to(self.DEVICE), lg.to(self.DEVICE)]))["out"].detach().cpu().numpy().flatten().tolist())
        
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
            case "automated_test":
                if len(cmpts) != 3:
                    print("Incorrect arguments, see \"help\"")
                else:
                    self.automated_test(cmpts[1], cmpts[2])
            case "slme":
                if len(cmpts) != 4:
                    if len(cmpts) != 2:
                        print("Incorrect arguments, see \"help\"")
                    else:
                        self.slme(cmpts[1])
                else:
                    self.slme(cmpts[1], float(cmpts[2]), float(cmpts[3]))
            case "set_models":
                if len(cmpts) != 3:
                    print("Incorrect arguments, see \"help\"")
                else:
                    self.set_models(cmpts[1], cmpts[2])
            case "set_data":
                if len(cmpts) != 2:
                    print("Incorrect arguments, see \"help\"")
                else:
                    self.set_data(cmpts[1])
            case _:
                print("Command not found, use \"help\"")

    def help(self):
        print(" <-- Help --> ")
        print(" Command List ({} are optional arguments):")
        print(Fore.RED + "  - help" + Fore.RESET)
        print("     Show this help info\n")
        print(Fore.RED + "  - quit" + Fore.RESET)
        print("     Quit the program\n")
        print(Fore.RED + "  - set_models [real directory] [imaginary directory]" + Fore.RESET)
        print("     Set the locations of the real and imaginary models\n")
        print(Fore.RED + "  - set_data [data directory]" + Fore.RESET)
        print("     Set the location of the data set to access\n")
        print(Fore.RED + "  - load_models" + Fore.RESET)
        print("     Load the checkpoints\n")
        print(Fore.RED + "  - plot [model_type] [ID]" + Fore.RESET)
        print("     Plot quickly with Plotille on terminal. model_type is either 'REAL' or 'IMAG'")
        print("     ID is the name of the structure file to evaluate for\n")
        print(Fore.RED + "  - slme [ID] [direct bandgap] [indirect bandgap]" + Fore.RESET)
        print("     Compute the SLME with structure \"ID\", and the bandgaps")
        print("     ID is the name of the structure file to evaluate for\n")
        print(Fore.RED + "  - automated_test [input text file] [output json]" + Fore.RESET)
        print("     Compute the SLME of all test structs and compute errors")
        print("     Prints the MAE and saves the computation information to output json\n")

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


def MAE(A, B):
    """Calculate the Mean Absolute Error of arrays A and B"""
    total = 0
    for i in range(0, len(A)):
        total += abs(A[i] - B[i])
    
    return total / len(A)


if __name__ == "__main__":
    parser = CommandParser()
    if len(sys.argv) == 1:
        parser.start()
    else:
        parser.parse_file(sys.argv[1])