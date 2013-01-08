# check_k.py

# Imports
import numpy as np
import argparse
from pprint import pprint

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('global_rotation_config', type=str)
    args = parser.parse_args()

    for key in ['global_rotation_config']:
        setattr(args, key, eval(getattr(args, key)))

    # parse `kg` to get the configuration of all of the global rotations
    kg = np.asarray(args.global_rotation_config)
    inst_info, basis_info = {}, {}
    k, i = 0, 0
    while i < kg.shape[0]:
        if kg[i] == 0:
            # fixed global rotation
            i += 1
        elif kg[i] < 0:
            # instance global rotation
            inst_info.setdefault(kg[i+1], []).append(k)
            i += 2
        else:
            # basis rotation
            n = kg[i]
            n_basis_info, n_coeff_info = basis_info.setdefault(n, ({}, {}))
            for j in xrange(n):
                n_basis_info.setdefault(kg[i+1+2*j], []).append(k)
                n_coeff_info.setdefault(kg[i+1+2*j+1], []).append(k)
            i += 2*n + 1

        k += 1

    # allocate global basis rotations
    pprint(inst_info)
    pprint(basis_info)

if __name__ == '__main__':
    main()
