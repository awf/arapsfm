# distance_map_generation.py

# Imports
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

# Test

# test_dt
def test_dt():
    import matplotlib.pyplot as plt
    from matplotlib import cm

    D = 6
    W = (D + 1) / 2
    O = W / 2

    # "inverted" segmentation
    A = np.ones((D,D), dtype=np.uint8)
    A[O:(O+W),O:(O+W)] = 0

    # colored template
    i = np.arange(A.size)
    np.random.shuffle(i)
    i = i.reshape(A.shape)
    cmap = cm.jet(np.linspace(0., 1., i.size, endpoint=True), 
                  bytes=True)[:,:-1]
    C = cmap[i]

    # distance transform
    dt_A, I = distance_transform_edt(A, return_indices=True)
    print 'arg_dt:', I
    print 'dt:', np.around(dt_A, decimals=2)
    arg_dt = C[tuple(I)]

    f, axs = plt.subplots(2,2)
    axs[0][0].imshow(A)
    axs[0][1].imshow(dt_A)
    axs[1][0].imshow(C)
    axs[1][1].imshow(arg_dt)
    plt.show()

# test_generate_distance_residuals
def test_generate_distance_residuals():
    import matplotlib.pyplot as plt
    from matplotlib import cm

    D = 50
    W = (D + 1) / 2
    O = W / 2

    # regular segmentation
    A = np.zeros((D,D), dtype=np.uint8)
    A[O:(O+W),O:(O+W)] = 1

    R = generate_distance_residuals(A)

    f, axs = plt.subplots(1,3)
    axs[0].imshow(A)
    axs[1].imshow(R[0])
    axs[2].imshow(R[1])
    plt.show()

# generate_distance_residuals
def generate_distance_residuals(mask):
    inverted_mask = np.ones_like(mask, dtype=np.uint8)
    inverted_mask[mask > 0] = 0

    dt, ind = distance_transform_edt(inverted_mask, 
                                     return_indices=True)

    r, c = np.mgrid[:inverted_mask.shape[0],
                    :inverted_mask.shape[1]]

    R = np.empty((2,) + inverted_mask.shape, dtype=np.float64)

    # for i in xrange(inverted_mask.size):
    #     # target row and column
    #     r, c = np.unravel_index(i, inverted_mask.shape)

    #     # individual residuals in (x, y)
    #     R[0, r, c] = c - ind[1, r, c]
    #     R[1, r, c] = -(r - ind[0, r, c])
    R[0, ...] = c - ind[1]
    R[1, ...] = -(r - ind[0])

    return R

if __name__ == '__main__':
    # test_dt()
    test_generate_distance_residuals()

