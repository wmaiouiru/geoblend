
import numpy as np
from scipy import sparse
import pyamg

from geoblend.vector import create_vector
from geoblend.coefficients import matrix_from_mask
from geoblend import blend


def test_blend_rectangular():

    mask = np.array([
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)

    source = np.array([
        [502, 527, 545, 517, 518, 492, 457, 562, 405, 420],
        [605, 512, 444, 473, 465, 496, 527, 445, 387, 397],
        [543, 446, 440, 393, 491, 472, 471, 417, 439, 371],
        [513, 476, 494, 448, 470, 491, 492, 443, 559, 514],
        [454, 487, 498, 471, 402, 484, 471, 377, 574, 452],
        [507, 478, 499, 484, 381, 372, 249, 333, 607, 410],
        [451, 497, 497, 392, 389, 476, 357, 366, 400, 464],
        [485, 517, 567, 531, 443, 324, 370, 408, 361, 464]
    ], dtype=np.float64)

    reference = np.array([
        [611, 573, 639, 564, 626, 588, 556, 503, 458, 461],
        [689, 559, 532, 550, 572, 601, 521, 466, 469, 437],
        [631, 530, 513, 504, 545, 516, 428, 444, 447, 430],
        [648, 566, 518, 514, 592, 537, 518, 468, 658, 559],
        [553, 587, 556, 544, 423, 574, 546, 452, 456, 387],
        [590, 598, 583, 564, 408, 389, 219, 498, 501, 479],
        [565, 572, 564, 436, 442, 638, 208, 382, 466, 455],
        [566, 545, 570, 507, 429, 378, 425, 474, 425, 466]
    ], dtype=np.float64)

    expected = np.array([
        [611, 573, 639, 564, 626, 588, 556, 503, 458, 461],
        [689, 584, 517, 540, 537, 563, 583, 475, 426, 437],
        [631, 523, 511, 459, 554, 530, 521, 457, 480, 430],
        [648, 558, 564, 510, 527, 543, 538, 483, 595, 559],
        [553, 561, 561, 526, 452, 531, 514, 413, 596, 387],
        [590, 543, 553, 529, 423, 416, 292, 373, 645, 479],
        [565, 556, 538, 422, 419, 517, 401, 410, 436, 455],
        [566, 545, 570, 507, 429, 378, 425, 474, 425, 466]
    ])
    
    # TODO: Encapsulate some of this logic in library
    m = matrix_from_mask(mask)
    v = np.ones((m.shape[0], 1))
    ml = pyamg.smoothed_aggregation_solver(m, v, max_coarse=10)

    indices = np.nonzero(mask)
    arr = blend(source, reference, mask, ml)
    arr[np.where(mask == 0)] = reference[np.where(mask == 0)]

    # Check that the numerical result is close to a precomputed result
    n = float(expected.size)
    y0, y1 = arr.min(), arr.max()
    err = np.sqrt(np.power(expected.astype(np.float) - arr.astype(np.float), 2).sum() / n) / (y1 - y0)

    assert err < 0.002


def test_blend():

    mask = np.array([
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
      [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
      [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)

    source = np.array([
        [502, 527, 545, 517, 518, 492, 457, 562, 405, 420],
        [605, 512, 444, 473, 465, 496, 527, 445, 387, 397],
        [543, 446, 440, 393, 491, 472, 471, 417, 439, 371],
        [513, 476, 494, 448, 470, 491, 492, 443, 559, 514],
        [454, 487, 498, 471, 402, 484, 471, 377, 574, 452],
        [507, 478, 499, 484, 381, 372, 249, 333, 607, 410],
        [451, 497, 497, 392, 389, 476, 357, 366, 400, 464],
        [485, 517, 567, 531, 443, 324, 370, 408, 361, 464]
    ], dtype=np.float64)

    reference = np.array([
        [611, 573, 639, 564, 626, 588, 556, 503, 458, 461],
        [689, 559, 532, 550, 572, 601, 521, 466, 469, 437],
        [631, 530, 513, 504, 545, 516, 428, 444, 447, 430],
        [648, 566, 518, 514, 592, 537, 518, 468, 658, 559],
        [553, 587, 556, 544, 423, 574, 546, 452, 456, 387],
        [590, 598, 583, 564, 408, 389, 219, 498, 501, 479],
        [565, 572, 564, 436, 442, 638, 208, 382, 466, 455],
        [566, 545, 570, 507, 429, 378, 425, 474, 425, 466]
    ], dtype=np.float64)

    expected = np.array([
        [611, 573, 639, 564, 626, 588, 556, 503, 458, 461],
        [689, 559, 513, 542, 572, 601, 521, 466, 469, 437],
        [631, 530, 508, 455, 550, 516, 428, 444, 447, 430],
        [648, 566, 558, 500, 509, 512, 495, 462, 658, 559],
        [553, 587, 556, 513, 429, 492, 463, 368, 556, 387],
        [590, 598, 583, 525, 397, 365, 215, 303, 501, 479],
        [565, 572, 564, 436, 394, 459, 208, 382, 466, 455],
        [566, 545, 570, 507, 429, 378, 425, 474, 425, 466]
    ])

    # TODO: Encapsulate some of this logic in library
    m = matrix_from_mask(mask)
    v = np.ones((m.shape[0], 1))
    ml = pyamg.smoothed_aggregation_solver(m, v, max_coarse=10)

    indices = np.nonzero(mask)
    arr = blend(source, reference, mask, ml)
    arr[np.where(mask == 0)] = reference[np.where(mask == 0)]

    # Check that the numerical result is close to a precomputed result
    n = float(expected.size)
    y0, y1 = arr.min(), arr.max()
    err = np.sqrt(np.power(expected.astype(np.float) - arr.astype(np.float), 2).sum() / n) / (y1 - y0)

    assert err < 0.0008
    