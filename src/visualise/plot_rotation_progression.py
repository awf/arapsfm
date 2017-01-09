# plot_rotation_progression.py

# Imports
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pprint import pprint

# Constants
CORE_FILENAME = 'core.npz'
CMAP = cm.RdYlBu

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    args = parser.parse_args()

    make_path = lambda f: os.path.join(args.input_dir, f)

    all_dirs = filter(lambda f: os.path.isdir(make_path(f)),
                      os.listdir(args.input_dir))

    def dir_id(dir_):
        dir_split = dir_.split('A')
        if len(dir_split) > 1:
            return 2 * int(dir_split[0])
        else:
            return 2 * int(dir_split[0]) + 1

    all_dirs = sorted(all_dirs, key=dir_id)
    core_paths = map(lambda d: make_path(os.path.join(d, CORE_FILENAME)),
                     all_dirs)
    ygs = map(lambda p: np.load(p)['yg'], core_paths)

    # NOTE Only for single-dimension basis rotation
    ygs = np.squeeze(ygs)
    l = np.linspace(0., 1., ygs.shape[0], endpoint=True)

    f, ax = plt.subplots()
    for i, yg in enumerate(ygs):
        ax.plot(yg, 'o-', c=CMAP(l[i]), label=all_dirs[i])
    ax.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()
