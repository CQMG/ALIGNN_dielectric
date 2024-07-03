#!/usr/bin/env python3

print("Loading, wait for \">\" prompt...")

import traceback, json, sys

import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.db.jsonutils import loadjson

import plotille, math
from colorama import Fore, Back, Style

import numpy as np

import matplotlib.pyplot as plt


