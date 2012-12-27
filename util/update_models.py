# update_models.py

# Imports
import os
import argparse
import numpy as np
from util.model_conversion import load_mesh, supported_ext

DATA_ROOT = 'data'
INPUT_DIR = os.path.join(DATA_ROOT, 'unconverted_models')
OUTPUT_DIR = os.path.join(DATA_ROOT, 'models')

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default=INPUT_DIR)
    parser.add_argument('-o', '--output', default=OUTPUT_DIR)
    args = parser.parse_args()

    if os.path.isdir(args.input):
        files = filter(lambda f: os.path.splitext(f)[1] in supported_ext(),
                       os.listdir(args.input))
        model_paths = map(lambda f: os.path.join(args.input, f), files)

        roots = map(lambda f: os.path.splitext(f)[0], files)
        output_paths = map(lambda f: os.path.join(args.output, r + '.npz'), 
                           roots)
    else:
        model_paths = [args.input]

        head, tail = os.path.split(args.output)
        output_root = os.path.splitext(tail)[0]
        output_paths = [os.path.join(head, output_root + '.npz')]

    output_dir = os.path.split(output_paths[0])[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_path, output_path in zip(model_paths, output_paths):
        print '<- %s' % input_path
        V, cells = load_mesh(input_path)

        print '-> %s' % output_path
        np.savez_compressed(output_path, points=V, cells=cells)

if __name__ == '__main__':
    main()


