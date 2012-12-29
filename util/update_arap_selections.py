# update_arap_selections.py

# Imports
import os
import numpy as np
import argparse
import operator
from pprint import pprint

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--ext', default='.npz', type=str)
    args = parser.parse_args()

    if os.path.isdir(args.input):
        files = filter(lambda f: os.path.splitext(f)[1] == args.ext,
                       os.listdir(args.input))
        args.input = map(lambda f: os.path.join(args.input, f), files)
    else:
        args.input = [args.input]

    def process_single(path):
        print '<- %s' % path
        z = np.load(path)

        k2 = []
        for k in z['K']:
            if k[0] == 0:
                k2.append([0])
            elif k[0] > 0:
                # NOTE Only 1-basis rotations can be interpreted from `z['K']`
                k2.append([1, k[1], k[0] - 1])
            elif k[0] < 0:
                k2.append([-1, k[1]])

        d = {k:z[k] for k in z.keys()}
        d['k2'] = np.asarray(reduce(operator.add, k2), dtype=np.intc)
        print '-> %s' % path
        np.savez_compressed(path, **d)

    map(process_single, args.input)
        
if __name__ == '__main__':
    main()
