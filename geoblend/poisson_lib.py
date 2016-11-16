import numpy as np
def convolve_mask_aware(arr, mask, threshold=0.0):
    """
    Convolve a 2D array with a 3x3 structuring element. The convolution operates only over
    regions specified by the mask. The following kernel is used:

     0  -1   0
    -1   4  -1
     0  -1   0

    :param arr:
        A 2D array that will undergo convolution.
    :param mask:
        ndarray where nonzero values represent the region
        of valid pixels in an image. The mask should be
        typed to uint8.
    :param threshold:
        Difference threshold to turn off the convolution approximation.
    """
    height = mask.shape[0]
    width = mask.shape[1]

    #i, j, nj, ni, sj, si, ej, ei, wj, wi, neighbors
    # x, s, d
    img = np.zeros((height, width), dtype=np.float64)
    assert arr.shape[0] == height
    assert arr.shape[1] == width

    for j in range(height):
        for i in range(width):

            if mask[j, i] == 0:
                continue

            x = 0.0
            s = arr[j, i]
            neighbors = 0

            # Define indices of 4-connected neighbors
            nj = (j - 1)
            ni = (i)

            sj = (j + 1)
            si = (i)

            ej = (j)
            ei = (i + 1)

            wj = (j)
            wi = (i - 1)

            # 1. Check that the neighbor index is within image bounds
            # 2. Check that the neighbor index is within the mask

            if (nj >= 0):
                if (mask[nj, ni] != 0):
                    neighbors += 1
                    x += (-1.0 * arr[nj, ni])
                else:
                    d = np.fabs(s - arr[nj, ni])
                    if (d > threshold):
                        neighbors += 1
                        x += (-1.0 * arr[nj, ni])

            if  (sj < height):
                if (mask[sj, si] != 0):
                    neighbors += 1
                    x += (-1.0 * arr[sj, si])
                else:
                    d = np.fabs(s - arr[sj, si])
                    if (d > threshold):
                        neighbors += 1
                        x += (-1.0 * arr[sj, si])

            if (ei < width):
                if (mask[ej, ei] != 0):
                    neighbors += 1
                    x += (-1.0 * arr[ej, ei])
                else:
                    d = np.fabs(s - arr[ej, ei])
                    if (d > threshold):
                        neighbors += 1
                        x += (-1.0 * arr[ej, ei])

            if (wi >= 0):
                if (mask[wj, wi] != 0):
                    neighbors += 1
                    x += (-1.0 * arr[wj, wi])
                else:
                    d = np.fabs(s - arr[wj, wi])
                    if (d > threshold):
                        neighbors += 1
                        x += (-1.0 * arr[wj, wi])

            x += (neighbors * arr[j, i])
            img[j, i] = x
    return np.asarray(img)

def create_vector_from_field(source, reference, mask):
    """
    Computes the column vector needed to solve the linearized Poisson equation.

    :param source:
        The vector field that will be preserved.
    :param reference:
        The reference image represented by a uint16 ndarray. This
        image will be used to sample for the boundary conditions.
    :param mask:
        ndarray where nonzero values represent the region
        of valid pixels in an image. The mask should be
        typed to uint8.
    """

    height = mask.shape[0]
    width = mask.shape[1]

    #cdef unsigned int i, j, nj, ni, sj, si, ej, ei, wj, wi, neighbors
    #cdef unsigned int
    idx = 0
    #cdef double coeff, s

    n = np.count_nonzero(mask)

    # PyAMG requires a double typed vector
    vector = np.empty(n, dtype=np.float64)

    assert source.shape[0] == height
    assert source.shape[1] == width
    assert reference.shape[0] == height
    assert reference.shape[1] == width

    for j in range(height):
        for i in range(width):

            if mask[j, i] == 0:
                continue

            neighbors = 0

            # Define indices of 4-connected neighbors
            nj = (j - 1)
            ni = (i)

            sj = (j + 1)
            si = (i)

            ej = (j)
            ei = (i + 1)

            wj = (j)
            wi = (i - 1)

            # Keep a running variable that represents the
            # element of the vector. This will be assigned
            # to the array at the end.
            coeff = 0.0
            s = source[j, i]

            if mask[nj, ni] == 0:
                coeff += (2 * reference[nj, ni])
            else:
                neighbors += 1

            if mask[sj, si] == 0:
                coeff += (2 * reference[sj, si])
            else:
                neighbors += 1

            if mask[ej, ei] == 0:
                coeff += (2 * reference[ej, ei])
            else:
                neighbors += 1

            if mask[wj, wi] == 0:
                coeff += (2 * reference[wj, wi])
            else:
                neighbors += 1

            coeff += (neighbors * s)

            # Assign the value to the output vector
            vector[idx] = coeff
            idx += 1
    return np.asarray(vector)
