# visualise_user_constraints.py

# Imports
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# inst_num
def inst_num(filename):
    root = os.path.splitext(filename)[0]
    try:
        return int(root.split('_')[-1])
    except ValueError:
        return None

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('image_stem')
    parser.add_argument('output_dir')
    parser.add_argument('--indices', type=str, default='None')
    parser.add_argument('--color', type=str, default='None')

    args = parser.parse_args()

    for key in ['indices', 'color']:
        setattr(args, key, eval(getattr(args, key)))

    pprint(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    make_input_path = lambda f: os.path.join(args.input_dir, f)
    make_output_path = lambda f: os.path.join(args.output_dir, f)
    load_image = lambda i: plt.imread(args.image_stem % i)

    input_files = filter(lambda f: os.path.isfile(make_input_path(f)), 
                         os.listdir(args.input_dir))

    valid_files = filter(lambda f: inst_num(f) is not None, input_files)
    sorted_files = sorted(valid_files, key=inst_num)

    if args.indices is not None:
        sorted_files = filter(lambda f: inst_num(f) in args.indices,
                              sorted_files)

    print '#:', len(sorted_files)

    for i, input_file in enumerate(sorted_files):
        n = inst_num(input_file)
        im = load_image(n)

        input_path = make_input_path(input_file)
        print ' <- (%d) %s' % (n, input_path)

        z = np.load(input_path)
        P = z['P']
        if hasattr(z, 'close'):
            z.close()

        P[:,1] = im.shape[0] - P[:,1]

        f = plt.figure(frameon=False)
        ax = f.add_axes([0., 0., 1., 1.])
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        x, y = np.transpose(P)
        ax.plot(x, y, 'ro', ms=8., c='w', mec='r', mew=3.)
        ax.set_xlim(0, im.shape[1] - 1)
        ax.set_ylim(im.shape[0] - 1, 0)

        w, h = im.shape[1], im.shape[0]
        dpi = 180.
        f.set_size_inches(w / dpi, h / dpi)
        f.set_dpi(dpi)

        full_path = make_output_path('%d.png' % n)
        f.savefig(full_path, dpi=dpi, bbox_inches='tight', pad_inches=0.)
        plt.close(f)

if __name__ == '__main__':
    main()

