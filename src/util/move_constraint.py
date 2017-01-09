# move_constraint.py

# Imports
import os
import numpy as np
import argparse

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('index', type=int)
    parser.add_argument('--new_index', type=int)
    args = parser.parse_args()

    full_paths = map(lambda f: os.path.join(args.input_dir, f),
                     os.listdir(args.input_dir))

    for full_path in full_paths:
        print '<- %s' % full_path
        z = np.load(full_path)
        d = {k:z[k] for k in ['T', 'V']}

        C = z['C'].tolist()
        P = z['P'].tolist()

        try:
            i = C.index(args.index)
        except ValueError:
            continue

        if args.new_index is None:
            del C[i]
            del P[i]
        else:
            C[i] = args.new_index

        C = np.asarray(C, dtype=np.int32)
        P = np.asarray(P, dtype=np.float64)

        d['C'] = d['point_ids'] = C
        d['P'] = d['positions'] = P

        print '%s ->' % full_path
        np.savez_compressed(full_path, **d)

if __name__ == '__main__':
    main()

