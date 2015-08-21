
# This file is called "b" since the function below returns the column
# vector traditionally known as "b" when solving a linear system Ax=b.

import numpy as np
cimport numpy as np


DTYPE_INT8 = np.int8
DTYPE_UINT8 = np.uint8
DTYPE_UINT16 = np.uint16
DTYPE_INT32 = np.int32

ctypedef np.int8_t DTYPE_INT8_t
ctypedef np.uint8_t DTYPE_UINT8_t
ctypedef np.uint16_t DTYPE_UINT16_t
ctypedef np.int32_t DTYPE_INT32_t


def b(np.ndarray[DTYPE_UINT8_t, ndim=2] mask, np.ndarray[DTYPE_INT32_t, ndim=2] field, np.ndarray[DTYPE_UINT16_t, ndim=2] reference):
    """
    Computes the column vector b from the linearized Poisson equation.
    
    :param mask:
        ndarray where nonzero values represent the region
        of valid pixels in an image. The mask should be
        typed to uint8.
    :param field:
        The guidance field that will be preserved during blending. Usually,
        this is the gradient of the source image, though it can be any other
        vector field.
    :param reference:
        The reference image that will be used to sample for the boundary
        conditions.
    
    .. todo:: The guidance field passed has thus far been an approximation of
              the gradient. A more precise representation may be calculated here
              if the source image is passed. Plus it will save the loop required
              when convolving the source image to find the gradient field.

    .. todo:: Can field be Int16?
    """
    
    assert mask.dtype == DTYPE_UINT8
    assert field.dtype == DTYPE_INT32
    assert reference.dtype == DTYPE_UINT16

    cdef int height = mask.shape[0]
    cdef int width = mask.shape[1]

    assert field.shape[0] == height
    assert field.shape[1] == width
    assert reference.shape[0] == height
    assert reference.shape[1] == width

    cdef int nj, ni, ej, ei, sj, si, wj, wi, bn
    cdef int idx = 0
    cdef int coeff

    # TODO: This value is also needed by matrix_from_mask. Loops can be
    #       reduced if needed, by sacrificing modularity and combining
    #       these functions.
    cdef int n = np.count_nonzero(mask)
    cdef np.ndarray[DTYPE_INT32_t, ndim=1] vector = np.zeros(n, dtype=np.int32)

    # TODO: nogil shiz?
    for j in range(height):
        for i in range(width):

            if mask[j][i] == 0:
                continue

            # Define indices of 4-connected neighbors
            nj, ni = j - 1, i
            sj, si = j + 1, i
            ej, ei = j, i + 1
            wj, wi = j, i - 1

            # Keep a running variable that represents the
            # element of the vector. This will be assigned
            # to the array at the end.

            # The major value is the value of the guidance field
            # at the index of the pixel.

            coeff = 8 * field[j][i]
            
            # Track the number of boundary neighbors
            bn = 0

            if nj >= 0 and nj <= height:
                
                if mask[nj][ni] == 0:
                    bn += 1
                    coeff += 2 * reference[nj][ni]
                else:
                    coeff -= 2 * field[nj][ni]

            if sj >= 0 and sj <= height:

                if mask[sj][si] == 0:
                    bn += 1
                    coeff += 2 * reference[sj][si]
                else:
                    coeff -= 2 * field[sj][si]

            if ei >= 0 and ei <= width:

                if mask[ej][ei] == 0:
                    bn += 1
                    coeff += 2 * reference[ej][ei]
                else:
                    coeff -= 2 * field[ej][ei]
            
            if wi >= 0 and wi <= width:

                if mask[wj][wi] == 0:
                    bn += 1
                    coeff += 2 * reference[wj][wi]
                else:
                    coeff -= 2 * field[wj][wi]
            
            if bn > 0:
                coeff -= ((2 * bn + 8) * reference[j][i])

            # Assign the value to the output vector
            vector[idx] = coeff
            idx += 1

    return vector
