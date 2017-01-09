# generate_distance_maps.py

# Imports
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from core_recovery.distance_map_generation import generate_distance_residuals

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('background_colour', type=str)
    parser.add_argument('--indices', type=str, default='None')
    parser.add_argument('--show_residual_maps', default=False, 
                        action='store_true')

    args = parser.parse_args()
    for key in ['background_colour', 'indices']:
        setattr(args, key, eval(getattr(args, key)))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    files = filter(lambda f: f.endswith('.png'), os.listdir(args.input_dir))
    file_roots = map(lambda f: os.path.splitext(f)[0], files)
    indices = map(lambda f: int(f.split('_')[-1]), file_roots)

    if args.indices is not None:
        def index_iterator():
            for i in args.indices:
                yield indices.index(i)
    else:
        def index_iterator():
            for i in np.argsort(indices):
                yield i

    for index in index_iterator():
        full_path = os.path.join(args.input_dir, files[index])
        print '<- %s' % full_path

        im = (plt.imread(full_path) * 255.).astype(np.uint8)
        mask = np.all(im != args.background_colour, axis=2)

        R = generate_distance_residuals(mask)

        if args.show_residual_maps:
            f, axs = plt.subplots(2,2)
            axs[0][0].imshow(mask)
            axs[0][1].imshow(np.sqrt(R[0]*R[0] + R[1]*R[1]))
            axs[1][0].imshow(R[0])
            axs[1][1].imshow(R[1])
            plt.show()

        output_path = os.path.join(args.output_dir, 
                                   file_roots[index] + '.npz')
        print '-> %s' % output_path
        np.savez_compressed(output_path, R=R)

if __name__ == '__main__':
    main()

