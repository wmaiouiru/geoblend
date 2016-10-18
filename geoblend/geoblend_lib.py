#!/usr/bin/env python

import os
import sys
import argparse
import logging

import pyamg
import numpy as np
import rasterio as rio
from scipy.signal import convolve2d
from scipy.ndimage import binary_erosion

from geoblend.coefficients import matrix_from_mask
from geoblend.convolve import convolve_mask_aware
from geoblend.vector import create_vector_from_field

logging.basicConfig(level=logging.INFO)


def main(argv):

    argument_sample='python '+os.path.realpath(__file__)
    parser = argparse.ArgumentParser(description=argument_sample)
    parser.add_argument('-src_file',dest='src_file',nargs='?',help='src_file',default=None)
    parser.add_argument('-ref_file',dest='ref_file',nargs='?',help='ref_file',default=None)
    parser.add_argument('-output_file',dest='output_file',nargs='?',help='output_file',default=None)
    parser.add_argument('-mask_file',dest='mask_file',nargs='?',help='mask_file',default=None)

    args = parser.parse_args()

    srcpath=args.src_file
    refpath=args.ref_file
    dstpath=args.output_file
    mask_file=args.mask_file

    # The idea is to blend the srcpath into the refpath

    if not srcpath and not refpath and not dstpath:
        print parser.print_help()
        return 0 
    if mask_file:
        mask_file=args.mask_file
        print 'not implemented'
        return 0
    else: 
        logging.info("Creating mask from {}".format(srcpath))

        with rio.open(srcpath) as src:
            r = src.read(1)

            mask = np.clip(r, 0, 1)
            mask[0, :] = 0
            mask[-1, :] = 0
            mask[:, 0] = 0
            mask[:, -1] = 0

            mask = binary_erosion(mask, structure=np.ones((10, 10))).astype(np.uint8)

    with rio.open(srcpath) as src, rio.open(refpath) as ref:

        m = matrix_from_mask(mask)
        x0 = np.ones((m.shape[0], 1))
        ml = pyamg.smoothed_aggregation_solver(m, x0, max_coarse=5)

        mask_profile = src.profile
        mask_profile.update(count=1)
        mskpath = os.path.join(os.path.dirname(srcpath), 'mask.jpg')
        with rio.open(mskpath, 'w', **mask_profile) as dst:
            dst.write_band(1, (255 * mask).astype(np.uint8))

        dst_profile = src.profile
        with rio.open(dstpath, 'w', **dst_profile) as dst:

            for bidx in range(1, 4):

                source = src.read(bidx).astype(np.uint16)
                reference = ref.read(bidx).astype(np.uint16)

                gradient = convolve_mask_aware(source, mask)
                vec = create_vector_from_field(gradient, reference, mask)
                pix = np.round(ml.solve(b=vec, x0=x0, tol=1e-03, accel='cg'))
                pix = np.clip(pix, 0, 255)

                reference[mask != 0] = pix
                dst.write_band(bidx, reference.astype(np.uint8))


if __name__ == '__main__':
    main(sys.argv[1:])

