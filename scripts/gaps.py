#!/usr/bin/env python3

print("Loading modules, wait for \">\" prompt...")

import os, sys, json, traceback

import torch

from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.db.jsonutils import loadjson

import math
from colorama import Fore, Back, Style

from sklearn.metrics.pairwise import cosine_similarity

class CommandParser:
    def __init__(self):
        '''Initialization sets a number of default state variables, most of which are used by load_model and the plotting commands.'''
        self.MODEL_LOC = "output_data/best_model.pt"
        self.CONFIG_LOC = "output_data/config.json"
        self.PLOT_LOC = "sample_plots/"
        self.TEST_RESULT_LOC = "test_results/"
        self.TEST_FILE_LOC = "test_files/"

        self.DEVICE = torch.device("cpu")

        self.MODEL = None

        try:
            f = open('./env_config.json')
            env_config = json.load(f)
        except FileNotFoundError:
            print("Could not find env_config.json. There are two possibilities:\n - You didn't use the templates to create this model.\n - This model was built using an older version of these scripts, in which case\n    you should use the legacy version of this script.")
            exit(1)

        try:
            self.POSCAR_LOC = env_config['data']
            self.ID_PROP_LOC = os.path.join(env_config['data'], "id_prop.csv")
        except AttributeError:
            print("Malformed env_config.json: Missing key 'data'")
            exit(1)
        
        try:
            self.PART = env_config['type']
        except AttributeError:
            print("Malformed env_config.json: Missing key 'type'")
            exit(1)
        
        try:
            id_prop_file = open(self.ID_PROP_LOC)
            self.ID_PROP_LINES = id_prop_file.readlines()
        except FileNotFoundError:
            print("WARNING: id_prop.csv not found. Any attempt to access reference data will fail.")
    
        id_prop_file = open(self.ID_PROP_LOC)
        self.ID_PROP_LINES = id_prop_file.readlines()

    def load_model(self):
        '''Loads the model from the configured state.'''
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        config = loadjson(self.CONFIG_LOC)
        self.MODEL = ALIGNNAtomWise(ALIGNNAtomWiseConfig(**config["model"]))

        self.MODEL.state_dict()
        self.MODEL.load_state_dict(torch.load(self.MODEL_LOC))

        self.MODEL.to(self.DEVICE)
        self.MODEL.eval()

        print("Model loaded successfully:")
        print("  Device:\t" + str(self.DEVICE))
        print("  Type:  \t" + str(self.PART))


    def get_sample_reference_name(self, lines, name):
        """
        Utility for getting the stored reference data by the name of the structure file
        """

        name_fixed = name.replace(",", "").replace("\"", "").replace("'", "").replace("\n", "")

        out = []

        for line in lines:
            elements = line.split(",")
            if elements[0] == name or elements[0] == name_fixed:
                for i in range(1, len(elements)):
                    out.append(float(elements[i]))
                return out

        raise ValueError("Name not found")

    def query_bandgaps(self, query):
        """
        Takes a query that is either the filename or ID of a material, and returns its
        band gaps as a 2-tuple of (direct, indirect).
        """

        ID_str = query.strip()

        try: # Attempt to open the requested material in the current working directory
            atoms = Atoms.from_poscar(ID_str)
        except FileNotFoundError:
            atoms = Atoms.from_poscar(self.POSCAR_LOC + ID_str)

        cvn = Spacegroup3D(atoms).conventional_standard_structure

        g, lg = Graph.atom_dgl_multigraph(atoms)

        result = list((self.MODEL([g.to(self.DEVICE), lg.to(self.DEVICE)]))['out'].detach().cpu().numpy().flatten().tolist())

        return (result[0], result[1])

    def ref_bandgaps(self, query):
        """
        Takes a query that is the filename of a material, and returns its
        reference band gaps as a 2-tuple of (direct, indirect) if it exists,
        else return None
        """

        if self.ID_PROP_LINES == None:
            return None

        query_fixed = query.replace(",", "").replace("\"", "").replace("'", "").replace("\n", "")

        for line in self.ID_PROP_LINES:
            elements = line.split(",")
            if elements[0] == query_fixed or elements[0] == query:
                return (float(elements[1]), float(elements[2]))

    
    def gaps(self, query):
        model_result = self.query_bandgaps(query)

        print("Model Results:")
        print(f"  Direct:   {model_result[0]}\n  Indirect: {model_result[1]}\n")

        ref_result = self.ref_bandgaps(query)

        if ref_result == None:
            print("No reference data")
        else:
            print("Reference Results:")
            print(f"  Direct:   {ref_result[0]}\n  Indirect: {ref_result[1]}\n")


    def automated_test(self, test_file, out_file=None):
        """
        Perform an automated test with the structure identifiers in test_file
        Optionally, output the results as JSON to out_file
        """

        predicted_dir = []
        predicted_indir = []

        real_dir = []
        real_indir = []

        predicted_delta = [] # predicted_dir - predicted_indir
        real_delta = [] # real_dir - real_indir

        with open(self.TEST_FILE_LOC + test_file) as f:
            test_structs = list(map(lambda x: x.strip(), f.readlines()))

        for struct in test_structs:
            model_gaps = self.query_bandgaps(struct)
            ref_gaps = self.ref_bandgaps(struct)

            predicted_dir.append(model_gaps[0])
            predicted_indir.append(model_gaps[1])

            real_dir.append(ref_gaps[0])
            real_indir.append(ref_gaps[1])

            predicted_delta.append(model_gaps[0] - model_gaps[1])
            real_delta.append(ref_gaps[0] - ref_gaps[1])

        mae_dir = MAE(predicted_dir, real_dir)
        mae_indir = MAE(predicted_indir, real_indir)

        mae_delta = MAE(predicted_delta, real_delta)

        print("Mean Absolute Error (eV):")
        print(f"  Direct:   {mae_dir}\n  Indirect: {mae_indir}\n")

        if out_file != None:
            print("Preparing output JSON...")

            raw_data = {}

            for i in range(0, len(test_structs)):
                raw_data[test_structs[i]] = {
                    "predicted_dir": predicted_dir[i],
                    "predicted_indir": predicted_indir[i],
                    "real_dir": real_dir[i],
                    "real_indir": real_indir[i],
                    "predicted_delta": predicted_delta[i],
                    "real_delta": real_delta[i]
                }

            out_dict = {
                "raw_data": raw_data,
                "mae_dir": mae_dir,
                "mae_indir": mae_indir,
                "mae_delta": mae_delta
            }

            print(f"Direct MAE:   {mae_dir}")
            print(f"Indirect MAE: {mae_indir}")
            print(f"Delta MAE:    {mae_delta}")

            try:
                with open(self.TEST_RESULT_LOC + out_file, "w") as f:
                    json.dump(out_dict, f, indent=4)
                print(f"Successfully saved to {self.TEST_RESULT_LOC + out_file}")
            except Exception:
                print("Could not write to JSON file")
                raise


    def parse(self, statement):
        """
        Processes user commands and dispatches the correct function with appropriate arguments
        """
        cmpts = statement.split(" ")

        match cmpts[0]:
            case "quit":
                exit()
            case "help":
                self.help()
            case "load_model":
                self.load_model()
            case "gaps":
                self.gaps(cmpts[1])
            case "automated_test":
                self.automated_test(cmpts[1], cmpts[2])

            case _:
                print("Command not found, use \"help\"")

    def help(self):
        """
        Prints help info for the user
        """
        print(" <-- Help --> ")
        print(" Command List ({} are optional arguments):")
        print(Fore.RED + "  - help" + Fore.RESET)
        print("     Show this help info\n")
        print(Fore.RED + "  - quit" + Fore.RESET)
        print("     Quit the program\n")
        print(Fore.RED + "  - load_model" + Fore.RESET)
        print("     Load the checkpoint from file\n")
        print(Fore.RED + "  - gaps" + Fore.RESET)
        print("     Fetch the bandgaps from the model, and compare with reference gaps if available\n")
        print(Fore.RED + "  - automated_test [test file] {output file}" + Fore.RESET)
        print("     Perform an automated test using \"\\n\"-delimited test file and optionally output results to JSON file\n")


    def start(self):
        """
        Entry point for the state-aware command parser
        """
        while True:
            try:
                statement = input("\n > ")
                self.parse(statement)
            except Exception as e:
                traceback.print_exc()
    
    def parse_file(self, file):
        """
        Run the lines of a file as if the user had entered them
        """
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

def MAD(A):
    """Calculate the Mean Absolute Deviation of array A"""
    mean = sum(A) / len(A)

    total = 0
    for a in A:
        total += abs(a - mean)
    
    return total / len(A)

def cos_sim(A, B):
    return cosine_similarity([A], [B])[0][0]

def discrete_derivative(A):
    out = []
    for i in range(0, len(A) - 2):
        out.append((A[i+2] - A[i])/0.3)

    return out

def cos_sim_derivative(A, B):
    A_d = discrete_derivative(A)
    B_d = discrete_derivative(B)

    return cos_sim(A_d, B_d)

def discrete_derivative_broad(A):
    out = []
    for i in range(0, len(A) - 10):
        out.append((A[i+10] - A[i])/0.5)

    return out

def cos_sim_derivative_broad(A, B):
    A_d = discrete_derivative_broad(A)
    B_d = discrete_derivative_broad(B)

    return cos_sim(A_d, B_d)


if __name__ == "__main__":
    parser = CommandParser()
    if len(sys.argv) == 1:
        parser.start()
    else:
        parser.parse_file(sys.argv[1])
