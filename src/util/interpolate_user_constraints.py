# interpolate_user_constraints.py

# Imports
import argparse, os
import numpy as np
from pprint import pprint

# load_constraints
def load_constraints(full_path):
    z = np.load(full_path)
    i = np.argsort(z['C'])
    return z['C'][i], z['P'][i], z['T'], z['V']
    
# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    files = filter(lambda f: f.endswith('.npz'), os.listdir(args.input_dir))
    file_roots = map(lambda f: os.path.splitext(f)[0], files)
    indices = sorted(map(lambda f: int(f), file_roots))

    input_full_path = lambda i: os.path.join(args.input_dir, '%d.npz' % i)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_full_path = lambda i: os.path.join(args.output_dir, '%d.npz' % i)

    for k, i in enumerate(indices[:-1]):
        j = indices[k+1]

        # skip consecutive frames
        n = j - i + 1
        if n == 2:
            continue

        # load constraints for both source frames
        Ci, Pi, Ti, Vi = load_constraints(input_full_path(i))
        Cj, Pj, Tj, Vj = load_constraints(input_full_path(j))

        if np.any(Ci != Cj):
            raise ValueError('C[%d] != C[%d]' % (i, j))

        # lineraly interpolate Pi -> Pj
        for l, t in enumerate(np.linspace(0., 1., n, endpoint=True)):
            P = Pi * (1. - t) + Pj * t
            T = Ti if l < (n - l) else Tj
            V = Vi if l < (n - l) else Vj
            np.savez_compressed(output_full_path(i + l),
                                point_ids=Ci,
                                positions=P,
                                V=V,
                                T=T,
                                C=Ci, 
                                P=P)
        
if __name__ == '__main__':
    main()

