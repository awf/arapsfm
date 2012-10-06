# generate_silhouette_info.py

# Imports
from util.cmdline import *

from core_recovery.silhouette_candidates import \
    generate_silhouette_candidate_info

# main
def main():
    parser = argparse.ArgumentParser(
        description='Solve single frame core covery problem')
    parser.add_argument('mesh', type=str)
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--step', type=float, default=40.)

    args = parser.parse_args()

    # load the geometry and input mesh 
    V = load_input_geometry(args.input)
    T = load_input_mesh(args.mesh)
    print 'V.shape:', V.shape
    print 'T.shape:', T.shape

    info = generate_silhouette_candidate_info(V, T, step=args.step, verbose=True)
    info.update(**args.__dict__)
    print 'info.keys():'
    pprint(info.keys())

    print '-> %s' % args.output
    pickle_.dump(args.output, info)

if __name__ == '__main__':
    main()

