#!/usr/bin/env python3

print("Loading, wait for \">\" prompt...")

import traceback, json, sys, os

import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.db.jsonutils import loadjson

import plotille, math
from colorama import Fore, Back, Style

import matplotlib.pyplot as plt

class CommandParser:
    def __init__(self):
        '''Initialization sets a number of default state variables, most of which are used by load_model and the plotting commands.'''
        self.ID_PROP_LOC = "structure_data/id_prop.csv"
        self.MODEL_LOC = "output_data/best_model.pt"
        self.CONFIG_LOC = "output_data/config.json"
        self.POSCAR_LOC = "structure_data/"
        self.PLOT_LOC = "sample_plots/"
        self.TEST_RESULT_LOC = "test_results/"
        self.TEST_FILE_LOC = "test_files/"

        self.NORM_DIV_LOC = "output_data/norm_div.txt"

        self.DEVICE = torch.device("cpu")

        self.MODEL = None

        self.PART = None
        cwd = os.getcwd().split("/")[-1]
        if "REAL" in cwd: # Attempt to use the current working directory to decide whether this is a real or imaginary plot
            self.PART = "REAL"
        if "IMAG" in cwd:
            self.PART = "IMAG"

        self.QUICKPLOT_TEMPLATE_PLOTILLE = "{}:blue:_:dashed {}:red:?" # Alter the quickplot template here
        self.QUICKPLOT_TEMPLATE_IMG = "{}:black:_:dashed {}:red:?"

        id_prop_file = open(self.ID_PROP_LOC)
        self.ID_PROP_LINES = id_prop_file.readlines()
    
    def set_id_prop(self, new):
        self.ID_PROP_LOC = new
    
    def set_model_loc(self, new):
        self.MODEL_LOC = new
    
    def set_config_loc(self, new):
        self.CONFIG_LOC = new
    
    def set_poscar_loc(self, new):
        self.POSCAR_LOC = new
    
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
    
    def get_plot_data(self, queries, verbose=False):
        """
        Takes a list of queries and returns a tuple that encodes proper plotting info for Plotille or Matplotlib
        Properly parses ? and queries the loaded model
        """
        print(queries)

        Ylist = []
        colors = []
        styles = []
        labels = []

        for query in queries:
            cmpts = query.split(":")

            if len(cmpts) > 2 and cmpts[2] == "?": # Detect the ? for querying the model
                if self.MODEL == None: # Check if the model is available
                    print("Must call load_model first")
                    return None
                ID = cmpts[0]
                color = cmpts[1]
                colors.append(color)
                try: # Detect if the ID is an integer and use that to select an index in the file
                    ID_int = int(ID)
                    ID_str = self.ID_PROP_LINES[ID_int].split(",")[0]
                except ValueError: # Otherwise, treat it as the filename
                    ID_str = ID
                try: # Attempt to open the requested model in the current working directory
                    atoms = Atoms.from_poscar(ID_str)
                except FileNotFoundError: # If it isn't there, look in the POSCAR_LOC, this allows the user to use POSCAR in current
                    atoms = Atoms.from_poscar(self.POSCAR_LOC + ID_str) # working directory or from a downloaded database elsewhere
                cvn = Spacegroup3D(atoms).conventional_standard_structure
                
                labels.append(ID_str + " (Predicted)") # Make it clear on the plot that this is a prediction
                if verbose: # Verbose is mostly for the automated_test
                    print("Processed " + ID_str + " (Predicted)")

                try: # Will detect and use a normalization divisor if it was used during data processing
                    f = open(self.NORM_DIV_LOC, 'r')
                    norm_div = float(f.readlines()[0])
                    f.close()
                    
                except Exception:
                    norm_div = 1.0

                # Create the atom and line graphs, then evaluate the model 
                g, lg = Graph.atom_dgl_multigraph(atoms)
                Ylist.append(list(map(lambda x: x * norm_div, (self.MODEL([g.to(self.DEVICE), lg.to(self.DEVICE)]))["out"].detach().cpu().numpy().flatten().tolist())))


            else: # Not querying the model, just pulling downloaded data
                ID = cmpts[0]
                color = cmpts[1]
                colors.append(color)
                try:
                    ID_int = int(ID)
                    label = self.ID_PROP_LINES[ID_int].split(",")[0]
                    labels.append(label)
                    if verbose: # Verbose is for automated_test
                        print("Processed " + label)
                    Ylist.append(self.get_sample_reference(self.ID_PROP_LINES, ID_int))
                except ValueError:
                    Ylist.append(self.get_sample_reference_name(self.ID_PROP_LINES, ID))
                    labels.append(ID)
                    if verbose: # Verbose is for automated_test
                        print("Processed " + ID)
            
            if len(cmpts) > 3: # Process the style information from the input query
                styles.append(cmpts[3])
            else:
                styles.append("solid")
            
        return (Ylist, colors, styles, labels)
    
    def automated_test(self, test_file, out_file=None):
        """
        Perform an automated test with the structure identifiers in test_file
        Optionally, output the results as JSON to out_file
        """
        f = open(self.TEST_FILE_LOC + test_file)
        test_structs = list(map(lambda x: x.strip(), f.readlines()))
        f.close()

        queries = []
        # This isn't the most elegant solution, but it reuses the get_plot_data code
        # A high-throughput implementation would be much different
        for struct in test_structs:
            queries.append(struct + ":blue")
            queries.append(struct + ":red:?")
        
        Ylist, colors, styles, labels = self.get_plot_data(queries, verbose=True)

        means = []
        maes = []
        mads = []
        scores = []

        best = 0
        worst = 0

        # Print and prep for JSON creation
        for i in range(0, int(len(Ylist) / 2)):
            real = Ylist[2 * i]
            predicted = Ylist[2 * i + 1]

            mean = sum(real) / len(real)
            mae = MAE(real, predicted)
            mad = MAD(real)
            score = mae / mad
            means.append(mean)
            maes.append(mae)
            mads.append(mad)
            scores.append(score)

            if score > scores[worst]:
                worst = i
            
            if score < scores[best]:
                best = i

            print(labels[2 * i] + ":")
            print(" Mean:    " + str(mean))
            print(" MAE:     " + str(mae))
            print(" MAD:     " + str(mad))
            print(" MAE:MAD: " + str(score) + "\n")
        
        # Print some aggregate statistics
        print("Average Values:")
        print(" Mean:    " + str(sum(means) / len(means)))
        print(" MAE:     " + str(sum(maes) / len(maes)))
        print(" MAD:     " + str(sum(mads) / len(mads)))
        print(" MAE:MAD: " + str(sum(scores) / len(scores)) + "\n")

        print("Worst Sample (" + labels[worst * 2] + "):")
        print(" Mean:    " + str(means[worst]))
        print(" MAE:     " + str(maes[worst]))
        print(" MAD:     " + str(mads[worst]))
        print(" MAE:MAD: " + str(scores[worst]) + "\n")

        print("Best Sample (" + labels[best * 2] + "):")
        print(" Mean:    " + str(means[best]))
        print(" MAE:     " + str(maes[best]))
        print(" MAD:     " + str(mads[best]))
        print(" MAE:MAD: " + str(scores[best]) + "\n")

        # Attempt to open and write to JSON
        # These keys can later be accessed by histogram and scatter plot commands
        # TODO: Collect additional information for further processing
        if out_file != None:
            print("Preparing output JSON...")

            out_dict = {}
            for i in range(0, int(len(labels) / 2)):
                out_dict[labels[2 * i]] = {
                    "mean": means[i],
                    "mae": maes[i],
                    "mad": mads[i],
                    "score": scores[i]
                }
            
            try:
                f = open(self.TEST_RESULT_LOC + out_file, "w")
                json.dump(out_dict, f, indent=4)
                f.close()
                print("Successfully saved to " + self.TEST_RESULT_LOC + out_file)
            except Exception:
                print("Could not write to JSON file")
                raise
            

    def plot_hist(self, in_file, out_file, element, bins=5):
        """
        Create a histogram plot for a JSON test output file
        """
        plt.switch_backend('agg')
        try:
            f = open(self.TEST_RESULT_LOC + in_file, 'r')
            data = json.load(f)
            f.close()
        except Exception:
            print("Could not open requested file.")
            traceback.print_exc()
            return None
        
        values = []
        for key in data.keys():
            values.append(data[key][element])
        
        plt.figure(dpi=600)

        plt.xlabel(element)
        plt.ylabel("Count")

        plt.hist(values, bins=bins)
        plt.savefig(self.TEST_RESULT_LOC + out_file)
        plt.close()
    
    def plot_scatter(self, in_file, out_file, elementX, elementY, elementC=None):
        """
        Create a scatter plot for a JSON test output file
        """
        plt.switch_backend('agg')
        try:
            f = open(self.TEST_RESULT_LOC + in_file, 'r')
            data = json.load(f)
            f.close()
        except Exception:
            print("Could not open requested file.")
            traceback.print_exc()
            return None
        
        xValues = []
        yValues = []
        cValues = []

        for key in data.keys():
            xValues.append(data[key][elementX])
            yValues.append(data[key][elementY])
            if elementC != None:
                cValues.append(data[key][elementC])

        figure, axes = plt.subplots()

        scatter = axes.scatter(xValues, yValues, c=cValues)
        figure.colorbar(scatter)
        
        plt.figure(dpi=600)

        plt.xlabel(elementX)
        plt.ylabel(elementY)

#        if len(cValues) != 0:
#            scatter = plt.scatter(xValues, yValues, c=cValues)
#        else:
#            scatter = plt.scatter(xValues, yValues)
#        
#        plt.legend(handles=scatter.legend_elements()[0], title=elementC)

        plt.savefig(self.TEST_RESULT_LOC + out_file)
        plt.close()

    def plot_img(self, queries):
        """
        Plot directly to an image with Matplotlib
        """
        plt.switch_backend('agg')
        X = []
        curr = 0
        for i in range(0, 300):
            X.append(curr)
            curr += 0.05
        
        filename = queries[0]
        queries_stripped = queries[1:]
        
        try:
            Ylist, colors, styles, labels = self.get_plot_data(queries_stripped)
        except TypeError:
            return None

        plt.figure(dpi=600)
        for i in range(0, len(Ylist)):
            plt.plot(X, Ylist[i], label=labels[i], linestyle=styles[i], color=colors[i])

        ylabel = "diel. function"
        if self.PART == "IMAG": # Attempt to intelligently select the correct axis title
            ylabel = "Imag. Part diel. function"
        if self.PART == "REAL":
            ylabel = "Real Part diel. function"
        
        plt.xlabel('Energy (eV)')
        plt.ylabel(ylabel)
        plt.legend()

        plt.savefig(self.PLOT_LOC + filename)
        plt.close()
    
    def plotille_plot(self, X, yValues, colors, labels):
        """
        Generate a terminal plot with Plotille
        """
        fig = plotille.Figure()
        fig.width = 120
        fig.height = 35
        fig.set_x_limits(min_=0, max_=15)
        
        yMax = 0
        yMin = 0
        for Y in yValues:
            yMax = max(yMax, max(Y))
            yMin = min(yMin, min(Y))
        
        fig.set_y_limits(min_=yMin, max_=math.ceil(yMax))

        for i in range(0, len(yValues)):
            fig.plot(X, yValues[i], lc=colors[i], label=labels[i])
        
        return fig.show(legend=True)
        
    def plot(self, queries):
        """
        Plot queries directly to the terminal with Plotille
        """
        X = []
        curr = 0
        for i in range(0, 300):
            X.append(curr)
            curr += 0.05
        try:
            Ylist, colors, styles, labels = self.get_plot_data(queries)
        except TypeError:
            return None

        print(self.plotille_plot(X, Ylist, colors, labels))
        print()

        # Generate some statistics for the plot
        if len(Ylist) == 1:
            mad = MAD(Ylist[0])
            print("Mean:\t" + str(sum(Ylist[0]) / len(Ylist[0])))
            print("MAD:\t" + str(mad))
        
        if len(Ylist) == 2:
            mae = MAE(Ylist[0], Ylist[1])
            mad = MAD(Ylist[0])
            print("Mean:\t\t" + str(sum(Ylist[0]) / len(Ylist[0])))
            print("MAE:\t\t" + str(mae))
            print("MAD:\t\t" + str(mad))
            print("MAE:MAD:\t" + str(mae/mad))
        
        print()
    

    def quickplot(self, query):
        """
        Uses the QUICKPLOT_TEMPLATE for Plotille to plot on terminal
        The query is just the name of the structure
        """
        self.plot(self.QUICKPLOT_TEMPLATE_PLOTILLE.format(query, query).split(" "))
    
    def quickplot_img(self, name, query):
        """
        Uses the QUICKPLOT_TEMPLATE for images to plot to an image
        The query is just the name of the structure and name is the location to save the png
        """
        self.plot_img((name + " " + self.QUICKPLOT_TEMPLATE_IMG.format(query, query)).split(" "))
    
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
            case "set_id_prop":
                self.set_id_prop(cmpts[1])
            case "set_model_loc":
                self.set_model_loc(cmpts[1])
            case "set_config_loc":
                self.set_config_loc(cmpts[1])
            case "set_poscar_loc":
                self.set_poscar_loc(cmpts[1])
            case "quickplot":
                self.quickplot(cmpts[1])
            case "plot":
                self.plot(cmpts[1:])
            case "quickplot_img":
                self.quickplot_img(cmpts[1], cmpts[2])
            case "plot_img":
                self.plot_img(cmpts[1:])
            case "automated_test":
                if len(cmpts) > 2:
                    self.automated_test(cmpts[1], cmpts[2])
                else:
                    self.automated_test(cmpts[1])
            case "plot_hist":
                if len(cmpts) > 4:
                    self.plot_hist(cmpts[1], cmpts[2], cmpts[3], int(cmpts[4]))
                else:
                    self.plot_hist(cmpts[1], cmpts[2], cmpts[3])
            case "plot_scatter":
                if len(cmpts) > 5:
                    self.plot_scatter(cmpts[1], cmpts[2], cmpts[3], cmpts[4], cmpts[5])
                else:
                    self.plot_scatter(cmpts[1], cmpts[2], cmpts[3], cmpts[4])
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
        print("     Load the checkpoint from file set by set_model_loc\n")
        print(Fore.RED + "  - plot [plot 1] {plot 2} {plot 3} ..." + Fore.RESET)
        print("     Plot on terminal with Plotille")
        print("     Plots in the format [structure file name or id_prop index]:[color]:{\"?\" to use model and predict}")
        print("     Color options: "+Fore.BLACK+"black, "+Fore.RED+"red,"+Fore.GREEN+" green,"+Fore.YELLOW+" yellow,"+Fore.BLUE+" blue,"+Fore.MAGENTA+" magenta,"+Fore.CYAN+" cyan,"+Fore.WHITE+" white"+Fore.RESET+"\n")
        print(Fore.RED + "  - plot_img [file name] [plot 1] {plot 2} {plot 3} ..." + Fore.RESET)
        print("     Plot directly to a png file with Matplotlib")
        print("     Plots in the format [structure file name or id_prop index]:[color]:{\"?\" to use model and predict}")
        print("     Color options: "+Fore.BLACK+"black, "+Fore.RED+"red,"+Fore.GREEN+" green,"+Fore.YELLOW+" yellow,"+Fore.BLUE+" blue,"+Fore.MAGENTA+" magenta,"+Fore.CYAN+" cyan,"+Fore.WHITE+" white"+Fore.RESET+"\n")
        print(Fore.RED + "  - automated_test [test file] {output file}" + Fore.RESET)
        print("     Perform an automated test using \"\\n\"-delimited test file and optionally output results to JSON file\n")
        print(Fore.RED + "  - plot_hist [json results] [output png] [value to plot] {# of bins}" + Fore.RESET)
        print("     Plot a histogram using a results file of the 'value to plot' value in the file\n")
        print(Fore.RED + "  - plot_scatter [input file] [output png] [X element] [Y element] {Color element}" + Fore.RESET)
        print("     Plot scatter plot with the given elements from the results file. Can use color element to plot a third axis as the shade of the points\n")
        print(Fore.RED + "  - set_id_prop [file name]" + Fore.RESET)
        print("     Set the id_prop.csv location\n")
        print(Fore.RED + "  - set_model_loc [file name]" + Fore.RESET)
        print("     Set the location of the checkpoint file to load\n")
        print(Fore.RED + "  - set_config_loc [file name]" + Fore.RESET)
        print("     Set the location of the config.json file to load the model with\n")
        print(Fore.RED + "  - set_poscar_loc [file name]" + Fore.RESET)
        print("     Set the directory where the POSCAR files are stored\n")

    def start(self):
        """
        Entry point for the state-aware command parser
        """
        while True:
            try:
                statement = input("\n> ")
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
        
            


    def get_sample_reference_name(self, lines, name):
        """
        Utility for getting the stored reference data by the name of the structure file
        """
        try:
            f = open(self.NORM_DIV_LOC, 'r')
            norm_div = float(f.readlines()[0])
            f.close()
        except Exception:
            norm_div = 1.0

        name_fixed = name.replace(",", "").replace("\"", "").replace("'", "").replace("\n", "")

        out = []

        for line in lines:
            elements = line.split(",")
            if elements[0] == name or elements[0] == name_fixed:
                for i in range(1, len(elements)):
                    out.append(float(elements[i]) * norm_div)
                return out

        raise ValueError("Name not found")

    def get_sample_reference(self, lines, index):
        """
        Utility for getting the stored reference data by line index in id_prop.csv
        """
        try:
            f = open(self.NORM_DIV_LOC, 'r')
            norm_div = float(f.readlines()[0])
            f.close()
        except Exception:
            norm_div = 1.0

        out = []

        line = lines[index]
        elements = line.split(",")
        for i in range(1, len(elements)):
            out.append(float(elements[i]) * norm_div)
        
        return out


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



if __name__ == "__main__":
    parser = CommandParser()
    if len(sys.argv) == 1:
        parser.start()
    else:
        parser.parse_file(sys.argv[1])