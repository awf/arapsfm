# view_overlay.py

# Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('segmentation', type=str)

    args = parser.parse_args()

    f, ax = plt.subplots()

    input_image = plt.imread(args.input)
    ax.imshow(input_image)

    seg = plt.imread(args.segmentation)
    mask = np.all(seg != (0, 0, 0), axis=-1).astype(np.int32)
    overlay = np.array([[  0, 0, 0,   0],
                        [255, 0, 0, 127]], dtype=np.uint8)[mask]

    ax.imshow(overlay)
    plt.show()

if __name__ == '__main__':
    main()
