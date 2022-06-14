# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created on Nov 12 2020
author: Tomasz Kacprzak
"""

import os, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil
from collections import OrderedDict
from deep_lss import utils_logging, utils_io, utils_weights_calc
from deep_lss.utils_weights_calc import cosmo_rename
from deep_lss.random import perm_sequence
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
    parser.add_argument('--dirpath_output', type=str, required=True, 
                        help='output dir for the results')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    parser.add_argument('--seed', type=int, required=False, default=850125, 
                        help='seed for the cosmo/sim/shell sequence')
    

    args = parser.parse_args(args)

    utils_logging.set_all_loggers_level(args.verbosity)

    # get absolute paths
    args.filename_config = utils_io.get_abs_path(args.filename_config)
    args.dirpath_output = utils_io.get_abs_path(args.dirpath_output)


    return args


def resources(args):
    
    reso = {'main_memory':2600,
            'main_nsimult':200,
            'main_time_per_index':1, # hours
            'main_scratch':6500,
            'merge_memory':64000,
            'merge_time':24,
            'pass':{'partition': 'hourly'}}
    
    return reso



def main(indices, args):

    # make tf records

    args = setup(args)
    np.random.seed(args.seed)
    ctx = utils_io.load_context(args.filename_config)

    # some problem variables
    n_cosmo = ctx.n_theta_cosmo
    n_sims = 12
    n_shells = ctx.shell_n
    
    # repeat settings
    n_fields_per_survey = ctx.n_fields_per_survey
    n_fields_avail = n_cosmo*n_sims*n_shells
    n_fields_per_cosmo = n_shells*n_sims
    n_surveys_total = np.ceil(n_fields_per_cosmo/n_fields_per_survey).astype(int) * n_cosmo
    n_fields_total = n_surveys_total*n_fields_per_survey
    n_surveys_per_file = n_surveys_total//n_cosmo
    n_files = n_surveys_total//n_surveys_per_file
    redshift_bins = ctx.load_redshift_bins
    n_downsample = ctx.shell_pix//ctx.shell_pix_use
    
    LOGGER.info(f'n_fields_per_survey={n_fields_per_survey}')
    LOGGER.info(f'n_fields_avail={n_fields_avail}')
    LOGGER.info(f'n_surveys_total={n_surveys_total}')
    LOGGER.info(f'n_fields_total={n_fields_total}')
    LOGGER.info(f'n_surveys_per_file={n_surveys_per_file}')
    LOGGER.info(f'n_files={n_files}')
    LOGGER.info(f'redshift_bins={redshift_bins}')
    LOGGER.info(f'n_downsample={n_downsample} {ctx.shell_pix}->{ctx.shell_pix_use}')

    # output dirs
    dirpath_out = args.dirpath_output
    if args.test:
        dirpath_out = args.dirpath_output + '_small'
        n_sims = 2
        LOGGER.warning('=============================> test')
    LOGGER.info(f'converting {dirpath_out}')

    seq = 'nontomo' if len(redshift_bins)==1 else 'tomo'
    dirpath_tfrecords = get_dirpath_tfrecords(args.dirpath_output, seq=seq, npix=ctx.shell_pix_use)

    utils_io.robust_makedirs(dirpath_tfrecords)
    LOGGER.info(f'storing in: {dirpath_tfrecords}')
    
    # sequence of parameters
    all_sims = np.array(list(itertools.product(range(n_shells), range(n_sims))))
    sim_seq = perm_sequence(elements=all_sims, length=n_fields_total)
    cos_seq = np.repeat(perm_sequence(elements=np.arange(n_cosmo), length=n_surveys_total), n_fields_per_survey)
    
    def read_files(ishe, isim, icos):

        cosmo_str, _ = cosmo_rename(ctx.theta_cosmo[icos], isim, ctx)
        filepath_in = get_filepath_kappa_maps(dirpath_out, cosmo_str)
        cosmo_vals = ctx.theta_priors.theta_cosmo[icos]

        with h5py.File(filepath_in, 'r') as f:
            X = np.array(f[str(ishe)])
            y = np.array([cosmo_vals['Omega_m'], cosmo_vals['sigma_8']], dtype=np.float32)

        # select redshift bins
        X = X[...,redshift_bins]

        Xt = tf.convert_to_tensor(X)
        yt = tf.convert_to_tensor(y)


        return Xt, yt

    def downsample(X):

        # batch, height, width, channels
        X = tf.transpose(X, perm=[2,0,1,3])
        X = tf.nn.avg_pool2d(X, ksize=2, strides=2, padding='VALID')
        X = tf.transpose(X, perm=[1,2,0,3])
        return X


    def finalize_file_processing(filepath_out_local, filepath_out):

        shutil.copyfile(filepath_out_local, filepath_out)
        os.remove(filepath_out_local)
        LOGGER.debug(f'wrote {filepath_out}')

        # read test
        dataset = tf.data.TFRecordDataset(filepath_out).map(parse_inverse)
        j = 0
        for X, y, c in dataset:
            LOGGER.debug(f'read j={j:<4d} test={X.shape} {y} c={c}') # print parsed example messages with restored arrays      
            j+=1
            

    def write_sequence(index):

        # write write shuffle sequence
        j = index * n_surveys_per_file * n_fields_per_survey

        filename_out = f'maps{index:04d}.tfrecords'
        filepath_out_local = os.path.join('/scratch/', filename_out)
        filepath_out = os.path.join(dirpath_tfrecords, filename_out)
        LOGGER.debug(f'===============> file {index}/{n_files} n_surveys_per_file={n_surveys_per_file} {filename_out}')
    
        with tf.io.TFRecordWriter(filepath_out_local) as file_writer:

            for i in LOGGER.progressbar(range(n_surveys_per_file), at_level='info', desc=f'writing fields {filename_out} ...'):

                elements = []
                for k in range(n_fields_per_survey):

                    ishe, isim = sim_seq[j]
                    icos = cos_seq[j]
                    Xt, yt = read_files(ishe, isim, icos)
                    if n_downsample > 1:
                        Xt = downsample(Xt)
                    elements.append((Xt, yt))
                    LOGGER.debug(f'          j={j: 6d} icos={icos:<4d} ishe={ishe:<4d} isim={isim:<4d} yt={yt} Xt.shape={Xt.shape}')
                    j+=1

                Xt = np.array([e[0] for e in elements]) 
                yt = np.array(elements[0][1]) 

                file_writer.write(parse_forward(Xt, yt, icos).SerializeToString())

        finalize_file_processing(filepath_out_local, filepath_out)
        

    # tfrecords file loop
    for index in indices:

        LOGGER.info(f'============== index {index}')

        write_sequence(index)

        yield index


def perm_sequence(elements, length):

    perms = [np.random.permutation(elements) for _ in range(int(np.ceil(length/len(elements))))]    
    return np.concatenate(perms)[:length]


# https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
def parse_forward(X, y, c):
    """define the dictionary -- the structure -- of our single example"""

    data = {
        'n_fields' : _int64_feature(X.shape[0]),
        'n_pix' : _int64_feature(X.shape[1]),
        'n_maps' : _int64_feature(X.shape[3]),
        'n_zbins' : _int64_feature(X.shape[4]),
        'n_y': _int64_feature(y.shape[0]),
        'c': _int64_feature(c),
        'X' : _bytes_feature(serialize_array(X)),
        'y' : _bytes_feature(serialize_array(y))
    }

    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out

def parse_inverse(element):
    """use the same structure as above; it's kinda an outline of the structure we now want to create"""
    data = {
      'n_fields': tf.io.FixedLenFeature([], tf.int64),
      'n_pix': tf.io.FixedLenFeature([], tf.int64),
      'n_maps': tf.io.FixedLenFeature([], tf.int64),
      'n_zbins': tf.io.FixedLenFeature([], tf.int64),
      'n_y': tf.io.FixedLenFeature([], tf.int64),
      'c': tf.io.FixedLenFeature([], tf.int64),
      'X' : tf.io.FixedLenFeature([], tf.string),
      'y': tf.io.FixedLenFeature([], tf.string),
    }


    content = tf.io.parse_single_example(element, data)
    X = tf.io.parse_tensor(content['X'], out_type=tf.float32)
    X = tf.reshape(X, shape=[content['n_fields'], content['n_pix'], content['n_pix'], content['n_maps'], content['n_zbins']])
    y = tf.io.parse_tensor(content['y'], out_type=tf.float32)
    y = tf.reshape(y, shape=[content['n_y']])
    c = content['c']
    return X, y, c

def get_parse_inverse_func(n_fields, n_pix, n_maps, n_zbins, n_y):


    data = {
      'n_fields': tf.io.FixedLenFeature([], tf.int64),
      'n_pix': tf.io.FixedLenFeature([], tf.int64),
      'n_maps': tf.io.FixedLenFeature([], tf.int64),
      'n_zbins': tf.io.FixedLenFeature([], tf.int64),
      'n_y': tf.io.FixedLenFeature([], tf.int64),
      'c': tf.io.FixedLenFeature([], tf.int64),
      'X' : tf.io.FixedLenFeature([], tf.string),
      'y': tf.io.FixedLenFeature([], tf.string),
    }

    @tf.function()
    def parse_inv(element):

        content = tf.io.parse_single_example(element, data)
        X = tf.io.parse_tensor(content['X'], out_type=tf.float32)
        X = tf.reshape(X, shape=[n_fields, n_pix, n_pix, n_maps, n_zbins])
        y = tf.io.parse_tensor(content['y'], out_type=tf.float32)
        y = tf.reshape(y, shape=[n_y])
        c = content['c']
        return X, y, c

    return parse_inv



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
