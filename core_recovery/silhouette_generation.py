# silhouette_generation.py

# Imports
import os
import numpy as np
from scipy import ndimage
from itertools import count

# Segmentation to silhouette

# segmentation_to_boundary
def segmentation_to_boundary(mask):
    vertical = np.array([[0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0]], dtype=np.uint8)

    horizontal = np.transpose(vertical)

    edge_image = ((ndimage.convolve(mask, vertical) == 2) | 
                  (ndimage.convolve(mask, horizontal) == 2))

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
    ordered_indices[n] = True
    n += 1

    while True:
        p = positions[i]

        for offset in offsets:

            p_offset = p + offset
            if not np.all((0 <= p_offset) & (p_offset < reverse_map.shape)):
                continue

            i = reverse_map[tuple(p_offset)]
            if i == -1:
                continue

            if used_positions[i]:
                continue
        
            used_positions[i] = True
            ordered_indices[n] = i
            n += 1

            break

        else:

            break

    return positions[ordered_indices[:n]]

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
    
