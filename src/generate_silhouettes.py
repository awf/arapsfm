# generate_silhouettes.py

# Imports
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from core_recovery.silhouette_generation import generate_silhouette

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('background_colour', type=str)
    parser.add_argument('--subsample', type=int, default=10)
    parser.add_argument('--indices', type=str, default='None')
    parser.add_argument('--show_silhouettes', default=False, 
                        action='store_true')
    parser.add_argument('--flip_normals', default=False, action='store_true')
    parser.add_argument('--output_silhouettes_dir', type=str, default=None)

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

        S, SN = generate_silhouette(mask, args.subsample, args.flip_normals)

        if args.show_silhouettes:
            f = plt.figure()
            ax = f.add_axes((0., 0., 1., 1.), frameon=False)
            ax.imshow(mask)
            ax.set_xticks([])
            ax.set_yticks([])

            l = 0.025 * max(im.shape[0], im.shape[1])

            for i in xrange(S.shape[0]):
                X = np.r_['0,2', S[i], S[i] + l*SN[i]]
                x, y = np.transpose(X.astype(np.int32))
                ax.plot(x, mask.shape[0] - y, 'r-')

            ax.set_title(full_path)

            if args.output_silhouettes_dir is None:
                plt.show()
            else:
                if not os.path.exists(args.output_silhouettes_dir):
                    os.makedirs(args.output_silhouettes_dir)

                full_path = os.path.join(args.output_silhouettes_dir, '%d.png' % indices[index])
                print ' (->) %s' % full_path

                f.savefig(full_path)
                plt.close(f)    

        output_path = os.path.join(args.output_dir, 
                                   file_roots[index] + '.npz')
        print '-> %s' % output_path
        np.savez_compressed(output_path, S=S, SN=SN)

if __name__ == '__main__':
    main()

