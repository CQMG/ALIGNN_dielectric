#!/usr/bin/env python3

from model import FunctionModel

from jarvis.core.atoms import Atoms

import sys


model = FunctionModel()

if len(sys.argv != 3):
    print("Must have:\nrun_test.py [test file] [test output]")
    exit(1)

f = open(sys.argv[1])
in_lines = f.readlines()
f.close()

