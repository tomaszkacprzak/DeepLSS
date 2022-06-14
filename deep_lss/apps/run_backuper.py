# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created on Nov 12 2020
author: Tomasz Kacprzak
"""

import os, warnings, argparse, logging, shutil
from deep_lss import utils_logging, utils_io, utils_arrays

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

def setup(args):

    description = 'Straightforward copy directory'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--src', type=str, required=True, 
                        help='source dir')
    parser.add_argument('--dst', type=str, required=True, 
                        help='destination dir')
    argk, _ = parser.parse_known_args(args)

    utils_logging.set_all_loggers_level(argk.verbosity)

    return argk


def resources(args):

    reso = {}
    reso['main_memory'] = 1000
    reso['main_time_per_index'] = 1 # hours
    reso['main_nproc'] = 1
    reso['main_ngpu'] = 1
    reso['pass'] = {'cluster':'gmerlin6', 'partition':'gpu-short'}
    
    return reso

    
def main(indices, args):

    def run_cmd(cmd):
        LOGGER.info(f'running {cmd}')
        os.system(cmd)
        LOGGER.info(f'done!')

    args = setup(args)
    utils_io.robust_makedirs(args.dst)
    LOGGER.info(f'copying {args.src} -> {args.dst}')

    cmd = f'rsync -avR {args.src}/./*/checkpoints_lssnet {args.dst}/'
    run_cmd(cmd)
    
    cmd = f'rsync -avR {args.src}/./*/lssnet_logs {args.dst}/'
    run_cmd(cmd)

    yield 0

