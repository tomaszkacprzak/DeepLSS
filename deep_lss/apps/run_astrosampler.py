# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created on Nov 12 2020
author: Tomasz Kacprzak
"""

import os, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil
from collections import OrderedDict
from deep_lss import utils_logging, utils_io, utils_weights_calc
from deep_lss.utils_weights_calc import cosmo_rename
from deep_lss.filenames import *
import tensorflow as tf

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

def setup(args):

    description = 'Make maps'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--filename_config', type=str, required=True, 
                        help='configuration yaml file')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    argk, _ = parser.parse_known_args(args)

    utils_logging.set_all_loggers_level(argk.verbosity)

    # get absolute paths
    argk.filename_config = utils_io.get_abs_path(argk.filename_config)

    return argk


def resources(args):

    res = dict(main_memory=2600,
                main_time_per_index=4, # hours
                main_scratch=6500,
                merge_memory=64000,
                merge_time=24)
    return res


def main(indices, args):

    args = setup(args)
    ctx = utils_io.load_context(args.filename_config)
    dirpath_out = ctx.files['astro_prior_grid']

    if not os.path.isdir(ctx.files['astro_prior_grid']):
        os.makedirs(dirpath_out)
        LOGGER.info(f'made dir {dirpath_out}')
    LOGGER.info(f'serializing files to {dirpath_out}')

    n_files = 16
    n_samples = n_files*int(1e6)
    n_dim = ctx.theta_priors.n_astro_params

    from sobol_seq import i4_sobol_generate
    LOGGER.info(f'getting {n_samples} sobol samples in {n_dim} dims')
    samples = i4_sobol_generate(dim_num=n_dim, n=n_samples).astype(np.float32)
    samples_split = np.array_split(samples, n_files)
    for i in range(n_files):

        filepath_out = os.path.join(dirpath_out, f'astrosamples{i:02d}.tfrecords')
        with tf.io.TFRecordWriter(filepath_out) as file_writer:
            for y in LOGGER.progressbar(samples_split[i], at_level='info', desc=f'serializing file {i}/{n_files}'):
                file_writer.write(parse_forward(y).SerializeToString())


    LOGGER.info('done')
    yield 0


def parse_forward(y):
    """define the dictionary -- the structure -- of our single example"""

    data = {
        'n_y': _int64_feature(y.shape[0]),
        'y' : _bytes_feature(serialize_array(y))
    }

    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out

@tf.function()
def parse_inverse(element):
    """use the same structure as above; it's kinda an outline of the structure we now want to create"""
    data = {
      'n_y': tf.io.FixedLenFeature([], tf.int64),
      'y': tf.io.FixedLenFeature([], tf.string),
    }

    content = tf.io.parse_single_example(element, data)
    y = tf.io.parse_tensor(content['y'], out_type=tf.float32)
    y = tf.reshape(y, shape=[content['n_y']])
    return y

def serialize_array(array):
    """Serialize tensor"""    
    array = tf.io.serialize_tensor(array)
    return array

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
