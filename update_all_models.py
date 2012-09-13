# update_all_models.py

# Imports
import os
import numpy as np
from util.model_conversion import load_mesh, supported_ext

DATA_ROOT = 'data'
INPUT_DIR = os.path.join(DATA_ROOT, 'unconverted_models')
OUTPUT_DIR = os.path.join(DATA_ROOT, 'models')

# main
def main():
    model_files = filter(lambda f: os.path.splitext(f)[1] in supported_ext(),
                         os.listdir(INPUT_DIR))

    for filename in model_files:
        full_path = os.path.join(INPUT_DIR, filename)
        print '<- %s' % full_path
        V, cells = load_mesh(full_path)

        root, ext = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_DIR, root + '.npz')
        print '-> %s' % output_path
        np.savez_compressed(output_path, points=V, cells=cells)

if __name__ == '__main__':
    main()


