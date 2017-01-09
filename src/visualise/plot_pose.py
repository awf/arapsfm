# plot_pose.py

# Imports
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.linalg import norm
from geometry import axis_angle
from pprint import pprint
from misc.pickle_ import dump

# Constants
CORE_FILENAME = 'core.npz'

# inst_num
def inst_num(filename):
    root = os.path.splitext(filename)[0]
    try:
        return int(root.split('_')[-1])
    except ValueError:
        return None

# del_rot
def del_rot(x0, x1):
    return axis_angle.axAdd(x1, axis_angle.axScale(-1.0, x0))

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    args = parser.parse_args()

    make_path = lambda f: os.path.join(args.input_dir, f)
    all_files = filter(lambda f: os.path.isfile(make_path(f)),
                       os.listdir(args.input_dir))
    instance_files = filter(lambda f: inst_num(f) is not None, all_files)
    instance_files = sorted(instance_files, key=inst_num)
    instance_paths = map(make_path, instance_files)

    get_item = lambda z, k: np.squeeze(z[k])
    get_pose = lambda z: (get_item(z, 'Xg'), get_item(z, 's'))
    Xgs, ss = zip(*map(lambda p: get_pose(np.load(p)), instance_paths))

    length_Xgs = map(norm, Xgs)

    del_Xgs = map(del_rot, Xgs[:-1], Xgs[1:])
    length_del_Xgs = map(norm, del_Xgs)

    # global pose
    f = plt.figure()
    axs = [f.add_subplot(311, projection='3d'),
           f.add_subplot(323),
           f.add_subplot(324),
           f.add_subplot(325),
           f.add_subplot(326),]
            
    Xgs = np.asarray(Xgs)

    min_, max_ = np.amin(Xgs, axis=0), np.amax(Xgs, axis=0)
    axs[0].set_xlim(min_[0] - 0.05, max_[0] + 0.05)
    axs[0].set_ylim(min_[1] - 0.05, max_[1] + 0.05)
    axs[0].set_zlim(min_[2] - 0.05, max_[2] + 0.05)
    x, y, z = np.transpose(Xgs)
    axs[0].plot(x, y, z, 'bo-')

    # absolute pose
    length_Xgs = np.asarray(length_Xgs)
    n = np.arange(length_Xgs.shape[0])
    axs[1].plot(n, np.rad2deg(length_Xgs), 'bo-')

    # change in pose
    length_del_Xgs = np.asarray(length_del_Xgs)
    n = np.arange(length_del_Xgs.shape[0])
    axs[2].plot(n, np.rad2deg(length_del_Xgs), 'bo-')

    # scale
    ss = np.asarray(ss)
    n = np.arange(ss.shape[0])
    axs[3].plot(n, ss, 'bo-')

    del_ss = ss[1:] - ss[:-1]
    axs[4].plot(n[:-1], del_ss, 'bo-')

    input_dir = (args.input_dir[:-1] if args.input_dir.endswith(os.path.sep)
                                     else args.input_dir)
    axs[0].set_title(input_dir.split(os.path.sep)[-1])
    plt.show()

if __name__ == '__main__':
    main()
