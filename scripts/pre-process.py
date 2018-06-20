#!/usr/bin/env python3

# Core imports
import argparse
import logging
import math
import os

# 3rd party imports
import cv2
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# These 2 methods are taken directly from:
# https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


def process(in_dir, out_dir):
    """
    Process all files in direcotry in_dir (assumes they're all images)
    based on MNIST pre-processing methodoligy to normalise them against
    the training dataset (to give neural nets trained on the published
    dataset a fighting chance of inference from them).
    """

    # Based on: https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
    logger.debug("Processing files from directory %s into directory %s",
        in_dir, out_dir)

    if not os.path.isdir(out_dir):
        # cv2.imwrite will **silently** fail if the output directory
        # doesn't exist
        raise FileNotFoundError("Output directory does not exist: %s" % out_dir)

    for file_ in os.scandir(in_dir):
        if not file_.is_file():
            logger.warn("Skipping '%s', not a plain file", file_.name)
            continue
        if file_.name.startswith('.'):
            logger.warn("Skipping '%s', hidden directory", file_.name)
            continue

        outfile = os.path.join(out_dir, file_.name)

        logger.info("Processing: %s -> %s", file_.path, outfile)

        # Load image
        gray = cv2.imread(file_.path, cv2.IMREAD_GRAYSCALE)



        # resize the images and invert it (black background)
        gray = cv2.resize(255-gray, (28, 28), interpolation=cv2.INTER_AREA)


        (thresh, gray) = cv2.threshold(gray, 128, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Remove blank rows/columns from edges
        while np.sum(gray[0]) == 0:
            gray = gray[1:]

        while np.sum(gray[:,0]) == 0:
            gray = np.delete(gray,0,1)

        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]

        while np.sum(gray[:,-1]) == 0:
            gray = np.delete(gray,-1,1)

        rows,cols = gray.shape

        # Resize to fit 20x20 box
        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            gray = cv2.resize(gray, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            gray = cv2.resize(gray, (cols, rows))

        # Turn it into a 28x28 box
        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

        shiftx,shifty = getBestShift(gray)
        shifted = shift(gray,shiftx,shifty)
        gray = shifted

        # save the processed images
        logger.debug("Writing %s", outfile)
        cv2.imwrite(outfile, gray)

if __name__ == '__main__':
    # Parse arguments
    argparser = argparse.ArgumentParser(
        description="MNIST test image pre-processor")

    # Logging options
    output_group=argparser.add_argument_group("output options")
    output_group.add_argument('-l', '--level',
        choices=['debug', 'info', 'warn', 'error'], default='info',
        help="Set minimum log level. (default: info)")
    output_group.add_argument('--prefix', action='store_true',
        help="Add log level to stderr logging output.")

    # Input/output directory
    arggrp=argparser.add_argument_group("Input/Output files")
    arggrp.add_argument('directory',
        help="Directory to process images from")
    arggrp.add_argument('--outdir',
        help="Directory to store processed images in (default 'processed')",
        default='processed')

    args=argparser.parse_args()
 
    # Map argument values to numeric log level
    logging_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARN,
        'error': logging.ERROR,
    }
 
    # Setup logging before anything else.
    fmt_normal="%(message)s"
    # Longest pre-defiend log level is 'CRITICAL', which is 8 characters.
    fmt_prefix="[ %(levelname)-8s ] %(message)s"
 
    # Log to stdout
    log_handler = logging.StreamHandler()
    if args.prefix:
        log_handler.setFormatter(logging.Formatter(fmt=fmt_prefix))
    else:
        log_handler.setFormatter(logging.Formatter(fmt=fmt_normal))
    logger.addHandler(log_handler)
 
    logger.setLevel(logging_levels[args.level])
 
    logger.debug("Logging initialised.")
    logger.debug("Command line arguments: %s", args)
    process(args.directory, args.outdir)
