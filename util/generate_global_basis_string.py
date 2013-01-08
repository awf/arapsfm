# generate_global_basis_string.py

# Imports
import argparse
from pprint import pprint
from itertools import count
from operator import add

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('basis_indices', type=str)
    parser.add_argument('num_instances', type=int)
    parser.add_argument('--start_coefficient', type=int, default=0)

    args = parser.parse_args()
    args.basis_indices = eval(args.basis_indices)

    coefficient_counter = count(args.start_coefficient)
    n = len(args.basis_indices)

    components = []
    for i in xrange(args.num_instances):
        component = [n]
        for j in xrange(n):
            component.append(args.basis_indices[j])
            component.append(next(coefficient_counter))

        components.append(list(component))
    
    all_components = reduce(add, components)
    component_string = str(all_components).replace(' ', '')
    print '"%s"' % component_string

if __name__ == '__main__':
    main()
