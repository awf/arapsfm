# cmdline.py

# Imports
import argparse

import numpy as np
from misc import pickle_

from os.path import splitext, split

from mesh import faces, geometry

from pprint import pprint

# Functions

# Utilities
def requires(args, *keys):
    for key in keys:
        attr = getattr(args, key)
        if attr is None:
            raise ValueError('argument "%s" is required' %
                             key)
    
# Loaders

# load
def load(full_path):
    head, input_file = split(full_path)
    root, ext = splitext(input_file)

    if ext == '.npz':
        z = np.load(full_path)
        return {k:z[k] for k in z.keys()}
    else:
        return pickle_.load(full_path)

# load_input_mesh
def load_input_mesh(input_file):
    # load triangles
    z = load(input_file)
    return faces.faces_from_cell_array(z['cells'])

# load_input_geometry
def load_input_geometry(input_file, use_linear_transformation=False):
    # load geometry
    z = load(input_file)
    V = np.asarray(z['V'], dtype=np.float64)

    # apply linear transformation if available
    if use_linear_transformation and 'T' in z.keys():
        print 'applying linear transformation:'
        print z['T']
        T = np.asarray(z['T'], dtype=np.float64)
        S = T[:3, :3]
        t = T[:3, -1]

        V = np.dot(V, np.transpose(S)) + t

    return V

# load_args
def load_args(input_file, *keys):
    z = load(input_file)
    return tuple(z[key] for key in keys)

# Parsers

# parse_float_string
def parse_float_string(float_str):
    floats = float_str.split(',')
    return np.array([float(f) for f in floats], dtype=np.float64)

# parse_solver_options
def parse_solver_options(solver_options, **kwargs):
    opts = kwargs.copy()
    if solver_options is not None:
        opts.update(eval(solver_options))

    return opts
