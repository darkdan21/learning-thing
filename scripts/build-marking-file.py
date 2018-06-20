#!/usr/bin/env python3

# core modules
import argparse
import logging

# 3rd party modules
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def load(file_, array):
	array.append(np.array(cv2.imread(file_, cv2.IMREAD_GRAYSCALE)))

def save(file_, array):
    logger.info("Writing %s", file_)
    np.save(file_, np.array(array, dtype=np.uint8))

if __name__ == '__main__':
	# Parse arguments
    argparser = argparse.ArgumentParser(
        description="Turn pre-processed images into a numpy file for testing")

    # Logging options
    output_group=argparser.add_argument_group("output options")
    output_group.add_argument('-l', '--level',
        choices=['debug', 'info', 'warn', 'error'], default='info',
        help="Set minimum log level. (default: info)")
    output_group.add_argument('--prefix', action='store_true',
        help="Add log level to stderr logging output.")

    # Input/output directory
    arggrp=argparser.add_argument_group("Input/Output files")
    arggrp.add_argument('file', nargs='+',
        help="File to add to the file (will be added in order)")
    arggrp.add_argument('--out', required=True,
        help="Output file to store images in")

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

    array = [] # Will hold an array of numpy arrays
    for file_ in args.file:
        logger.debug("Processing: %s", file_)
        load(file_, array)

    save(args.out, array)
