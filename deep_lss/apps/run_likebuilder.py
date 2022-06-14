# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created on Nov 12 2020
author: Tomasz Kacprzak
"""

import os, warnings, argparse, h5py, numpy as np, time, itertools, tensorflow as tf
from collections import OrderedDict
from deep_lss import utils_logging, utils_io, utils_arrays
from deep_lss.configs import params as params_sims
from deep_lss.priors import convert_S8_to_sigma8, convert_sigma8_to_S8
from deep_lss.filenames import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

def setup(args):

    description = 'Train the LikeMDN likelihood model'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--filename_config', type=str, required=True, 
                        help='configuration yaml file')
    parser.add_argument('--dir_output', type=str, required=True, 
                        help='output dir')
    parser.add_argument('--models_filter', type=str, required=False, default=None,
                        help='only use these models')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    parser.add_argument('--draw', action='store_true',
                        help='only draw samples from pre-trained model')
    argk, _ = parser.parse_known_args(args)

    utils_logging.set_all_loggers_level(argk.verbosity)

    # get absolute paths
    argk.filename_config = utils_io.get_abs_path(argk.filename_config)
    argk.dir_output = utils_io.get_abs_path(argk.dir_output)
    utils_io.robust_makedirs(argk.dir_output)

    # make dirs
    utils_io.robust_makedirs(get_dirpath_checkpoints_likenet(argk.dir_output))

    # make TF log dirs
    utils_io.robust_makedirs(get_dirpath_likenet_logs(argk.dir_output))

    return argk


def resources(args):
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=str, default='gwen_short', choices=('gwen_short', 'gwen', 'gpu', 'gpu_short', 'cpu'))  
    argk, _ = parser.parse_known_args(args)
    reso = {}

    if argk.queue == 'gwen_short':

        reso['main_memory'] = 3900
        reso['main_time_per_index']=1 # hours
        reso['main_nproc'] = 16
        reso['main_ngpu']=1

        reso['merge_memory'] = 3900
        reso['merge_time'] = 1
        reso['merge_nproc'] = 1
        reso['merge_ngpu']=1
        
        reso['pass'] = {'cluster':'gmerlin6', 'account':'gwendolen', 'partition':'gwendolen'}

    elif argk.queue == 'gpu_short':

        reso['main_memory'] = 4000
        reso['main_time_per_index']=2 # hours
        reso['main_nproc'] = 4
        reso['main_ngpu']=1
        reso['merge_memory'] = 2600
        reso['merge_time'] = 1
        reso['merge_nproc'] = 4
        reso['pass'] = {'partition':'gpu-short', 'gpus-per-task':1, 'cpus-per-gpu':4}

    return reso

def main(indices, args):

    args = setup(args)
    ctx = utils_io.load_context(args.filename_config, models_filter=args.models_filter)
    utils_io.test_tensorflow()
    dir_predictions = get_dirpath_predictions(args.dir_output)

    for index in indices:
        LOGGER.timer.reset()

        run_likenet_training(ctx=ctx, 
                             ind_net=index,
                             file_pred_shuffled=get_filepath_predictions(dir_predictions, tag='shuffled'),
                             file_pred_repeated=get_filepath_predictions(dir_predictions, tag='repeated'),
                             dir_output=args.dir_output,
                             models_filter=args.models_filter,
                             n_samp_per_cosmo=1000,
                             test=args.test,
                             draw_only=args.draw)
        
        LOGGER.critical(f'=================> likenet training - done in {LOGGER.timer.elapsed()}')

    yield 0



def run_likenet_training(ctx, ind_net, file_pred_shuffled, file_pred_repeated, dir_output, n_samp_per_cosmo=100, models_filter=None, test=False, draw_only=False):
    """
    Train network that models the likelihood p(theta_est|theta_true)
    """

    # problem variables
    n_epochs=ctx.likelihood_modelling['n_epochs']
    batch_size=ctx.likelihood_modelling['batch_size']

    if test:
        n_epochs = 1
        n_samp_per_cosmo = 2
        LOGGER.warning('===================> test')


    # load data
    nets_y_true, nets_y_pred = utils_io.load_lssnet_predictions(file_pred_shuffled,
                                                                squeeze=True,
                                                                flatten=True,
                                                                models_filter=models_filter)

    netname_current = ctx.netnames[ind_net]
    y_true = nets_y_true[netname_current]
    y_pred = nets_y_pred[netname_current]

    y_true = utils_arrays.rec_to_arr(y_true, dtype=np.float32)
    y_pred = utils_arrays.rec_to_arr(y_pred, dtype=np.float32)

    LOGGER.info(f'training likenet for model={netname_current} with dataset y_pred={y_pred.shape}, y_true={y_true.shape}, n_gaussians={ctx.likelihood_modelling["likenet"]["n_gaussians"]}')

    select = np.all(np.isfinite(y_true), axis=1) & np.all(np.isfinite(y_pred), axis=1)
    y_true = y_true[select,:]
    y_pred = y_pred[select,:]
    LOGGER.info(f'removed nans/infs {np.count_nonzero(~select)}/{len(select)}')
    LOGGER.info(f'number of unique parameter combinations {len(np.unique(y_true.sum(axis=1)))}')

    # output
    dirpath_checkpoints = get_dirpath_checkpoints_likenet(dir_output)
    filepath_check = get_checkpoint_name(dirpath_checkpoints, 'best',  tag=f'likemdn__{netname_current}')


    # get model
    from deep_lss.networks.likemdn import LikeMDN
    net = LikeMDN(nx=y_true.shape[1],
                  ny=y_pred.shape[1],
                  filename_checkpoint=filepath_check,
                  **ctx.likelihood_modelling['likenet'])

    # tf logging
    dirpath_logs = get_dirpath_likenet_logs(dir_output)
    writer = tf.summary.create_file_writer(dirpath_logs)

    if draw_only:

        LOGGER.warning('====> draw only from pre-trained model')

    else:
    
        LOGGER.info(f'fitting mdn with {n_epochs} epochs with batch size {batch_size}')
        with writer.as_default():

            # fit
            net.fit(x=y_true, 
                    y=y_pred,  
                    epochs=n_epochs,
                    batch_size=batch_size,
                    verb=2)

        # store
        net.save(filepath_check)


    net.load(filepath_check)

    # load data
    nets_y_true, nets_y_pred = utils_io.load_lssnet_predictions(file_pred_repeated, models_filter=models_filter)

    y_true_rec = nets_y_true[netname_current]
    y_pred_rec = nets_y_pred[netname_current]
    y_true = utils_arrays.rec_to_arr(y_true_rec, dtype=np.float32)

    y_samp = net.sample(y_true, n_samples=1)
    y_samp_rec = utils_arrays.arr_to_rec(y_samp, cols=y_true_rec.dtype.names, dtype=np.float32)

    # store
    utils_io.store_predictions(get_dirpath_predictions(dir_output),
                               y_pred={netname_current: y_pred_rec},
                               y_true={netname_current: y_true_rec},
                               y_samp={netname_current: y_samp_rec},
                               tag=f'likenet_verif__{netname_current}')





def merge(indices, args):

    args = setup(args)
    ctx = utils_io.load_context(args.filename_config, models_filter=args.models_filter)
    cmd = f"esub repos/deep_lss/deep_lss/apps/run_constrainer.py --filename_config={args.filename_config} --dir_output={args.dir_output} --models_filter={args.models_filter} --tasks='0>400' --function=main+merge --n_cores=400 --mode=jobarray --job_name=con_{ctx.tag}"
    LOGGER.info(f'current dir: {os.getcwd()}')
    LOGGER.info('submitting command')
    print(cmd)
    from esub import esub
    import shlex
    args = shlex.split(cmd)[1:]
    ids = esub.main(args)
    LOGGER.info('done! hope it works..')














