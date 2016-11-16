#!/usr/bin/env python

import os
import sys
import argparse
import logging
import time
import pyamg
import numpy as np
import rasterio as rio
from scipy.signal import convolve2d
from scipy.ndimage import binary_erosion

from geoblend.coefficients import matrix_from_mask
from geoblend.convolve import convolve_mask_aware
from geoblend.vector import create_vector_from_field

import poisson_lib
logging.basicConfig(level=logging.INFO)
UINT8 = 255
UINT16 = 65535
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
            #mask[0, :] = 0
            #mask[-1, :] = 0
            #mask[:, 0] = 0
            #mask[:, -1] = 0

            mask_eroded = binary_erosion(mask, structure=np.ones((10, 10))).astype(np.uint8)
            print 'Created mask from channel'
    with rio.open(srcpath) as src, rio.open(refpath) as ref:
        m = matrix_from_mask(mask_eroded)
        x0 = np.ones((m.shape[0], 1))
        ml = pyamg.smoothed_aggregation_solver(m, x0, max_coarse=5)
        print 'smoothed_aggregation_solver initialized'
        # writing mask image for illustration purposes
        mask_profile = src.profile
        mask_profile.update(count=1)
        print src.meta
        print src.profile

        dst_meta = src.meta
        output_dtype = None
        clipping_value = None
        if src.meta['dtype'] == 'uint16':
            output_dtype = np.uint16
            clipping_value = UINT16
            mskpath = os.path.join(os.path.dirname(srcpath), 'mask.tif')
            dst_meta.update(count=3, compress='none', photometric='rgb', alpha='no')
        elif src.meta['dtype'] == 'uint8':
            output_dtype = np.uint8
            clipping_value = UINT8
            mskpath = os.path.join(os.path.dirname(srcpath), 'mask.jpg')
        else :
            print 'OUTPUT TYPE NOT SUPPORTED!'
            return

        with rio.open(mskpath, 'w', **mask_profile) as dst:
            dst.write_band(1, (clipping_value * mask_eroded).astype(output_dtype))
        print 'wrote mask file for illustration purpose'

        print 'looping each channel and and write possion blended output'
        with rio.open(dstpath, 'w', **dst_meta) as dst:
            # Iterate through each band
            for bidx in range(1, src.meta['count']+1):
                print 'writing band', bidx
                source = src.read(bidx).astype(np.uint16)
                reference = ref.read(bidx).astype(np.uint16)
                # combine src and ref where ref is zero
                ref_mask = np.clip(reference, 0, 1)
                reference[(mask-ref_mask)==1] = source[(mask-ref_mask)==1]
                print 'convert image to the gradient domain'
                #gradient = convolve_mask_aware(source, mask)
                current_time = time.time()
                gradient = poisson_lib.convolve_mask_aware(source, mask)
                print 'created gradient ',time.time() - current_time
                #vec = create_vector_from_field(gradient, reference, mask_eroded)

                current_time = time.time()
                vec = poisson_lib.create_vector_from_field(gradient, reference, mask_eroded)
                print 'created vec ',time.time() - current_time

                print 'solving the Poisson equation'
                current_time = time.time()
                pix = np.round(ml.solve(b=vec, x0=x0, tol=1e-03, accel='cg'))
                print 'saolved Poisson equation',time.time() - current_time

                pix = np.clip(pix, 0, clipping_value)
                reference[mask_eroded != 0] = pix
                dst.write_band(bidx, reference.astype(output_dtype))

if __name__ == '__main__':
    main(sys.argv[1:])
