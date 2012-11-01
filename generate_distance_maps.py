# generate_distance_maps.py

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from core_recovery.distance_map_generation import generate_distance_residuals

# Change `DATA_SOURCE` and `BACKGROUND_COLOR` for different sources

# Constants
DATA_ROOT = 'data'

# DATA_SOURCE = 'cheetah0'
# BACKGROUND_COLOUR = np.array([255, 242, 0, 255], dtype=np.uint8)

# DATA_SOURCE = 'circle'
# BACKGROUND_COLOUR = np.array([255, 242, 0], dtype=np.uint8)

# DATA_SOURCE = 'cheetah1'
# BACKGROUND_COLOUR = np.array([255, 255, 0], dtype=np.uint8)

DATA_SOURCE = 'cheetah1B'
BACKGROUND_COLOUR = np.array([255, 255, 0], dtype=np.uint8)

INPUT_DIR = os.path.join(DATA_ROOT, 'segmentations', DATA_SOURCE)
OUTPUT_DIR = os.path.join(DATA_ROOT, 'distance_maps', DATA_SOURCE)

# Options
# SHOW_RESIDUAL_MAPS = True
SHOW_RESIDUAL_MAPS = False

# load_inverse_segmentation
def load_inverse_segmentation(index):
    path = os.path.join(INPUT_DIR, '%d-INV_S.png' % index)
    print '<- %s' % path

    color_mask = (plt.imread(path)*255.).astype(np.uint8)
    return np.any(color_mask[..., :3] != BACKGROUND_COLOUR, axis=-1)

# save_distance_residuals
def save_distance_residuals(index, R):
    path = os.path.join(OUTPUT_DIR, '%d_D.npz' % index)
    print '-> %s' % path
    np.savez_compressed(path, R=R)

# main
def main():
    try:
        os.makedirs(OUTPUT_DIR)
    except os.error:
        # directory already exists
        pass

    all_files = os.listdir(INPUT_DIR)

    inv_seg_files = filter(lambda f: '-INV_S.png' in f, all_files)
    indices = map(lambda f: int(f.split('-INV_S')[0]), inv_seg_files)
    print 'indices:', indices

    for index in indices:
        mask = load_inverse_segmentation(index)
        R = generate_distance_residuals(mask)

        save_distance_residuals(index, R)

        if SHOW_RESIDUAL_MAPS:
            f, axs = plt.subplots(2,2)
            axs[0][0].imshow(mask)
            axs[0][1].imshow(np.sqrt(R[0]*R[0] + R[1]*R[1]))
            axs[1][0].imshow(R[0])
            axs[1][1].imshow(R[1])
            plt.show()

if __name__ == '__main__':
    main()

