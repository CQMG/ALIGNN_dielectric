# ALIGNN_dielectric
A set of utilities for the application of ALIGNN to materials' dielectric functions

# What's Included
This repository contains two main components:

- The training template folder
- The analyze.py script for analyzing models interactively

# Usage
To train a model, first follow the instructions for setting up ALIGNN at [ALIGNN's Github](https://github.com/usnistgov/alignn). Then, clone this repository to an accessible directory. To begin training, copy the train_template folder, which contains several scripts that prepare for and execute a proper training of the model.

Once the model has been trained, you can then cd into the training directory and run the analyze.py script, which will allow you to load a model, calculate and plot its results, run a battery of automated tests, and more. For more information about this script, run it and type `help`
