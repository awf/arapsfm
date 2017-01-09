# propagate_user_constraints.py

# Imports
import os, argparse
import numpy as np
from matop.nd_blocks import nd_blocks
import matplotlib.pyplot as plt
from pprint import pprint
import shutil
import heapq

# normalised_blocks
def normalised_blocks(im, I, patch_size):
    dim = (patch_size, patch_size)
    blocks = np.r_['1,2',
                   nd_blocks(im[...,0], I, dim),
                   nd_blocks(im[...,1], I, dim),
                   nd_blocks(im[...,2], I, dim)].astype(np.float64)

    return (blocks / (255.0 * np.product(dim)))

# load_image
def load_image(image_path):
    return (plt.imread(image_path) * 255.).astype(np.uint8)
    
# load_constraints
def load_constraints(constraints_path, image_path, patch_size, 
                     aux_keys=('T', 'V')):
    # load the constraints and the auxiliary keys
    z = np.load(constraints_path)
    i = np.argsort(z['C'])
    C = z['C'][i]
    P = z['P'][i]

    aux = {k:z[k] for k in aux_keys}

    # convert P -> I [(x,y) -> (r, c)]
    im = load_image(image_path)
    I = np.fliplr(np.around(P)).astype(np.int)
    I[:,0] = im.shape[0] - I[:,0]

    # load the appearance block at the desired positions
    A = normalised_blocks(im, I, patch_size)

    return C, P, A, aux

# new_constraint_positions
def new_constraint_positions(P, A, image_path, patch_size, n):
    # convert P -> I [(x,y) -> (r, c)]
    im = load_image(image_path)
    I = np.fliplr(np.around(P)).astype(np.int)
    I[:,0] = im.shape[0] - I[:,0]

    # offsets to test
    offsets = np.mgrid[-n:(n+1), -n:(n+1)].transpose(1, 2, 0).reshape(-1, 2)

    # fill the new (r, c) coordinates
    J = np.empty_like(I)

    for l, i in enumerate(I):
        r = normalised_blocks(im, i + offsets, patch_size) - A[l]
        d = np.sum(r*r, axis=-1)
        offset_index = np.argmin(d)

        J[l] = i + offsets[offset_index]

    J[:,0] = im.shape[0] - J[:,0]

    return np.require(np.fliplr(J), dtype=np.float64, requirements='C')

# propagate_single
def propagate_single(input_constraints, input_image, 
                     target_image, output_constraints,
                     patch_size, window_size, show_positions=False):

    C, P, A, aux = load_constraints(input_constraints,
                                    input_image,
                                    patch_size)

    Q = new_constraint_positions(P, A, target_image, 
                                 patch_size,
                                 window_size)

    print '-> %s' % output_constraints
    np.savez_compressed(output_constraints,
                        point_ids=C, positions=Q,
                        C=C, P=Q,
                        **aux)

    if show_positions:
        def plot_axis(ax, im, P, *args, **kwargs):
            ax.imshow(im)

            P_ = P.copy()
            P_[:,1] = im.shape[0] - P_[:,1]
            x, y = np.transpose(P_)
            ax.plot(x, y, *args, **kwargs)
            for i, c in enumerate(C):
                ax.text(P_[i,0], P_[i,1], '%d' % c)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, im.shape[1])
            ax.set_ylim(im.shape[0], 0)

        f = plt.figure(frameon=False)
        axs = [f.add_axes([0., 0.5, 1., 0.5]),
               f.add_axes([0., 0., 1., 0.5])]

        plot_axis(axs[0], load_image(input_image), P, 'ro')
        plot_axis(axs[1], load_image(target_image), Q, 'ro')
        plt.show()

# test_propagate_single
def test_propagate_single():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_constraints', type=str)
    parser.add_argument('input_image', type=str)
    parser.add_argument('target_image', type=str)
    parser.add_argument('output_constraints', type=str)
    parser.add_argument('--patch_size', type=int, default=5)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--show_positions', default=False, 
                        action='store_true')

    pprint(args)

    propagate_single(args.input_constraints,
                     args.input_image,
                     args.target_image,
                     args.output_path,
                     args.patch_size, 
                     args.window_size, 
                     args.show_positions)

# main_dir
def main_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('image_stem', type=str)
    parser.add_argument('--patch_size', type=int, default=5)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--show_positions', default=False, 
                        action='store_true')
    parser.add_argument('--visualise_only', type=str, default='None')

    args = parser.parse_args()
    for key in ['visualise_only']:
        setattr(args, key, eval(getattr(args, key)))

    pprint(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    input_full_path = lambda i: os.path.join(args.input_dir, '%d.npz' % i)
    output_full_path = lambda i: os.path.join(args.output_dir, '%d.npz' % i)

    files = filter(lambda f: f.endswith('.npz'), os.listdir(args.input_dir))
    file_roots = map(lambda f: os.path.splitext(f)[0], files)
    indices = np.asarray(sorted(map(lambda f: int(f), file_roots)),
                         dtype=np.int32)


    # propagate sequentially always going from closest user-specified
    # constraints
    _I = indices[0]
    to_process = []
    already_done = np.zeros(indices[-1] - indices[0] + 1, dtype=bool)

    for i in indices:
        src = input_full_path(i)
        dst = output_full_path(i)
        print '%s -> %s' % (src, dst)
        shutil.copy(src, dst) 

        heapq.heappush(to_process, (0, i))
        already_done[i - _I] = True

    while to_process:
        depth, i = heapq.heappop(to_process)

        for offset in (-1, 1):
            j = i + offset
            if not (indices[0] <= j and j <= indices[-1]):
                continue

            if already_done[j - _I]:
                continue

            print '%d -> %d:' % (i, j)

            # propagate
            src_constraints = output_full_path(i)
            src_image = args.image_stem % i
            dst_image = args.image_stem % j
            dst_constraints = output_full_path(j)
            print ' %s (%s) ->\n %s (%s)' % (src_constraints, src_image, 
                                             dst_constraints, dst_image)

            show_position = args.show_positions
            if show_position and args.visualise_only is not None:
                show_position = show_position and i in args.visualise_only

            propagate_single(src_constraints, src_image, 
                             dst_image, dst_constraints,
                             args.patch_size,
                             args.window_size,
                             show_position)

            already_done[j - _I] = True
            heapq.heappush(to_process, (depth + 1, j))

if __name__ == '__main__':
    main_dir()

