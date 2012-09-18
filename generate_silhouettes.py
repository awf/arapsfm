# generate_silhouettes.py

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from core_recovery.silhouette_generation import generate_silhouette

# Change `DATA_SOURCE` and `BACKGROUND_COLOR` for different sources

# Constants
DATA_ROOT = 'data'

# DATA_SOURCE = 'cheetah0'
DATA_SOURCE = 'circle'

# BACKGROUND_COLOUR = np.array([255, 242, 0, 255], dtype=np.uint8)
BACKGROUND_COLOUR = np.array([255, 242, 0], dtype=np.uint8)

INPUT_DIR = os.path.join(DATA_ROOT, 'segmentations', DATA_SOURCE)
OUTPUT_DIR = os.path.join(DATA_ROOT, 'silhouettes', DATA_SOURCE)

# Options
SUBSAMPLE = 20
FLIP_NORMALS = True
SHOW_SILHOUETTES = True

# Segmentation to silhouette

# load_inverse_segmentation
def load_inverse_segmentation(index):
    path = os.path.join(INPUT_DIR, '%d-INV_S.png' % index)
    print '<- %s' % path
    color_mask = (plt.imread(path)*255.).astype(np.uint8)
    return np.any(color_mask != BACKGROUND_COLOUR, axis=-1)

# save_silhouette
def save_silhouette(index, S, SN):
    path = os.path.join(OUTPUT_DIR, '%d_S.npz' % index)
    print '-> %s' % path
    np.savez_compressed(path, S=S, SN=SN)

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
        S, SN = generate_silhouette(mask,
                                    SUBSAMPLE,
                                    FLIP_NORMALS)
        save_silhouette(index, S, SN)
    
        if SHOW_SILHOUETTES:
            f = plt.figure()
            ax = f.add_axes((0., 0., 1., 1.), frameon=False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(mask)

            for i in xrange(S.shape[0]):
                X = np.vstack([S[i], S[i] + 15*SN[i]])
                x_, y = np.transpose(X.astype(int))

                ax.plot(x_, mask.shape[0] - y, 'r-')

            ax.set_title('Index: %d' % index)
            plt.show()


if __name__ == '__main__':
    main()

