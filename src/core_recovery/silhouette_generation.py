# silhouette_generation.py

# Imports
import os
import numpy as np
from scipy import ndimage
from itertools import count

DEBUG = False

if DEBUG:
    import matplotlib.pyplot as plt

# Segmentation to silhouette

# segmentation_to_boundary
def segmentation_to_boundary(mask):
    vertical = np.array([[0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0]], dtype=np.uint8)

    horizontal = np.transpose(vertical)

    edge_image = ((ndimage.convolve(mask, vertical) == 2) | 
                  (ndimage.convolve(mask, horizontal) == 2) &
                  (mask != 0))

    if DEBUG:
        offsets = np.array([[ 0, -1],
                            [ 1, -1],
                            [ 1,  0],
                            [ 1,  1],
                            [ 0,  1],
                            [-1,  1],
                            [-1,  0],
                            [-1, -1]], dtype=np.int32)

        i = np.argwhere(edge_image)
        neighbour_count = np.zeros_like(edge_image, dtype=np.int)

        for offset in offsets:
            j = i + offset
            j = j[np.all((j >= 0) & (j < edge_image.shape), axis=1)]
            j = j[edge_image[tuple(np.transpose(j))] != 0]
            neighbour_count[tuple(np.transpose(j))] += 1

        f, axs = plt.subplots(1, 2)
        axs[0].imshow(neighbour_count, cmap='jet')
        axs[1].imshow(mask)
        plt.show()

    return edge_image

# ordered_boundary
def ordered_boundary(edge_image):
    offsets = np.array([[ 0, -1],
                        [ 1, -1],
                        [ 1,  0],
                        [ 1,  1],
                        [ 0,  1],
                        [-1,  1],
                        [-1,  0],
                        [-1, -1]], dtype=np.int32)

    positions = np.argwhere(edge_image)
    N = positions.shape[0]

    used_positions = np.zeros(N, dtype=bool)

    reverse_map = np.empty(edge_image.shape, dtype=np.int32)
    reverse_map.fill(-1)
    reverse_map[tuple(np.transpose(positions))] = xrange(N)

    ordered_indices = np.empty(N, dtype=np.int32)

    i, n = 0, 0
    used_positions[i] = True
    ordered_indices[n] = 0
    n += 1

    longest_path = None

    while True:
        if DEBUG:
            z = np.zeros(reverse_map.shape + (4,), dtype=np.uint8)

            q = np.transpose(positions[ordered_indices[:n]])
            z[tuple(q) + (slice(None), )] = (255, 0, 0, 255)

            f, ax = plt.subplots()
            ax.imshow(edge_image)
            ax.imshow(z)

            j = np.argwhere(z)
            min_ = np.amin(j, axis=0) 
            max_ = np.amax(j, axis=0)
            ax.set_xlim(min_[1] - 5, max_[1] + 5)
            ax.set_ylim(max_[0] + 5, min_[0] - 5)

            plt.show()

        # current position
        p = positions[i]

        # progress to unused child
        has_progressed = False

        for offset in offsets:
            # potential child
            p_offset = p + offset

            # in bounds?
            if not np.all((0 <= p_offset) & (p_offset < reverse_map.shape)):
                continue

            # is an edgel pixel?
            i = reverse_map[tuple(p_offset)]
            if i == -1:
                continue

            # is used?
            if used_positions[i]:
                continue
        
            # progress
            used_positions[i] = True
            ordered_indices[n] = i
            n += 1

            has_progressed = True
            break

        if has_progressed:
            continue

        # not progressed, so save current path
        if longest_path is None or len(longest_path) < n:
            longest_path = ordered_indices[:n].copy()

        # backtrack
        if n > 0:
            n -= 1
            i = ordered_indices[n]
        else:
            break
            
    return positions[longest_path]

# ordered_boundary_normals
def ordered_boundary_normals(H, ordered_boundary, flip_normals=False):
    x = np.fliplr(ordered_boundary)
    x[:,1] = H - x[:,1]

    fd = np.r_[np.atleast_2d(x[1] - x[-1]),
               x[2:] - x[:-2],
               np.atleast_2d(x[0] - x[-2])]

    n = np.fliplr(fd).astype(np.float64)
    n[:,0] *= -1.0
    n /= np.sqrt(np.sum(n**2, axis=1))[:, np.newaxis]

    if flip_normals:
        n *= -1

    return x.astype(np.float64), n

# generate_silhouette
def generate_silhouette(mask, subsample=1, flip_normals=False):
    edge_image = segmentation_to_boundary(mask)

    ordered_edge = ordered_boundary(edge_image)
    N = ordered_edge.shape[0]
    
    sampled_edge = ordered_edge[::subsample]

    if N % subsample != 0:
        sampled_edge = sampled_edge[:-1]

    x, n = ordered_boundary_normals(edge_image.shape[0], 
                                    sampled_edge,
                                    flip_normals=flip_normals)

    return x, n
    
