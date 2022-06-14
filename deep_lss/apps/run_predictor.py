# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created on Nov 12 2020
author: Tomasz Kacprzak
"""

import os, warnings, argparse, h5py, numpy as np, time, itertools, importlib, tensorflow as tf
from collections import OrderedDict
from deep_lss import utils_logging, utils_io, utils_arrays
from deep_lss.configs import params as params_sims
from deep_lss.filenames import *
from deep_lss.apps.run_recordmaker import parse_inverse
from deep_lss.random import perm_sequence

# tensorflow config
gpus = tf.config.experimental.list_physical_devices('GPU')

# warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

def setup(args):

    description = 'Train DeepLSS networks'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--filename_config', type=str, required=True, 
                        help='configuration yaml file')
    parser.add_argument('--dir_output', type=str, required=True, 
                        help='output dir')
    parser.add_argument('--models_filter', type=str, required=False, default=None,
                        help='only use these models')
    parser.add_argument('--queue', type=str, default='gwen_short', choices=('gwen_short', 'gwen', 'gpu', 'gpu_short', 'cpu'),
                        help='which cluster configuration to use')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    parser.add_argument('--copy_dataset', action='store_true',
                        help='copy dataset')
    parser.add_argument('--sensitivity', action='store_true',
                        help='if to make sensitivity maps')
    argk, _ = parser.parse_known_args(args)

    utils_logging.set_all_loggers_level(argk.verbosity)

    # get absolute paths
    # argk.filename_config = utils_io.get_abs_path(argk.filename_config)
    # argk.dir_output = utils_io.get_abs_path(argk.dir_output)
    utils_io.robust_makedirs(argk.dir_output)

    # make dirs
    utils_io.robust_makedirs(get_dirpath_predictions(argk.dir_output))

    # make TF log dirs
    utils_io.robust_makedirs(get_dirpath_likenet_logs(argk.dir_output))

    if argk.test:
        tf.config.run_functions_eagerly(True)
        LOGGER.warning('========> test, running functions eagerly')

    return argk


def resources(args):

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=str, default='gwen_short', choices=('gwen_short', 'gwen', 'gpu', 'gpu_short', 'cpu'))  
    argk, _ = parser.parse_known_args(args)
    reso = {}    

    if argk.queue == 'cpu':

        reso['main_memory'] = 6400
        reso['main_time_per_index'] = 1 # hours
        reso['main_nproc'] = 1
        reso['main_scratch'] = 6500
        reso['main_ngpu'] = 0
        reso['merge_memory'] = 2600
        reso['merge_time'] = 0.5
        reso['pass'] = {'cluster':'merlin6', 'partition':'hourly'}

    elif argk.queue == 'gwen_short':

        reso['main_memory'] = 3900
        reso['main_time_per_index']=1 # hours
        reso['main_nproc'] = 16
        reso['main_ngpu']=1
        reso['merge_memory'] = 2600
        reso['merge_time'] = 1
        reso['merge_nproc'] = 16
        reso['pass'] = {'cluster':'gmerlin6', 'account':'gwendolen', 'partition':'gwendolen', 'gpus-per-task':1, 'cpus-per-gpu':16}
    
    elif argk.queue == 'gpu_short':

        reso['main_memory'] = 6000
        reso['main_time_per_index'] = 2 # hours
        reso['main_nproc'] = 5
        reso['main_ngpu'] = 1
        reso['merge_memory'] = 3900
        reso['merge_time'] = 1
        reso['merge_nproc'] = 5
        reso['pass'] = {'cluster':'gmerlin6', 'partition':'gpu-short', 'gpus-per-task':1, 'cpus-per-gpu':5}

    return reso

def main(indices, args):

    args = setup(args)
    ctx = utils_io.load_context(args.filename_config, args.models_filter)
    utils_io.test_tensorflow()
    seq = 'nontomo' if len(ctx.load_redshift_bins)==1 else 'tomo' 
    dirpath_maps = get_dirpath_tfrecords(ctx.files['projected_maps'], seq=seq,  npix=ctx.shell_pix_use)
    dir_output_preds = get_dirpath_predictions(args.dir_output)


    # load networks
    nets = utils_io.get_networks(ctx=ctx,
                                 dirpath_checkpoints=get_dirpath_checkpoints(args.dir_output),
                                 epoch='last')

    for index in indices:
    
        with tf.device('gpu'):

            # seed generator
            g = tf.random.Generator.from_seed(index*2313)
            tf.random.set_global_generator(g)


        if not args.sensitivity:
    
            LOGGER.critical('predictions for repeated theta - starting')
            LOGGER.timer.reset()
            y_pred_shuffled, y_true_shuffled, y_pred_repeated, y_true_repeated, y_pred_fiducial, y_true_fiducial = get_lssnet_predictions(ctx=ctx,
                                                      nets=nets,
                                                      path_maps=dirpath_maps,
                                                      index=index, 
                                                      test=args.test,
                                                      sensitivity=False)

            file_pred_shuffled = utils_io.store_predictions(dir_output_preds, y_pred_shuffled, y_true_shuffled, tag=f'shuffled_index{index:04d}')
            file_pred_repeated = utils_io.store_predictions(dir_output_preds, y_pred_repeated, y_true_repeated, tag=f'repeated_index{index:04d}')
            file_pred_fiducial = utils_io.store_predictions(dir_output_preds, y_pred_fiducial, y_true_fiducial, tag=f'fiducial_index{index:04d}')
            LOGGER.critical(f'predictions - done in {LOGGER.timer.elapsed()}')

    
        else:

            LOGGER.critical('getting sensitivity maps only')
            nets_sensitivity = get_lssnet_predictions(ctx=ctx,
                                                      nets=nets,
                                                      path_maps=dirpath_maps,
                                                      index=index, 
                                                      test=args.test,
                                                      sensitivity=True)

            file_sensitivity_maps = get_filepath_sensitivity_maps(dir_output_preds, tag=f'index{index:02d}')
            utils_io.write_to_pickle(file_sensitivity_maps, nets_sensitivity)

        yield index




def merge(indices, args):

    # here merge 

    args = setup(args)
    ctx = utils_io.load_context(args.filename_config, args.models_filter)

    LOGGER.critical('merging indices')
    LOGGER.timer.reset()

    collect_prediction_results(indices=indices, 
                               dir_output=get_dirpath_predictions(args.dir_output))

    LOGGER.critical(f'merging - done in {LOGGER.timer.elapsed()}')
            
    # for index in indices:
    #     filepath_pred = get_filepath_predictions(dir_output, tag=f'{seq}_index{index:04d}')
    #     if os.path.isfile(filepath_pred):
    #         os.remove(filepath_pred)


def get_lssnet_predictions(ctx, nets, path_maps, cpu=tf.data.AUTOTUNE, index=0, test=False, sensitivity=False):
    """
    get predictions 
    """

    # get problem dimensions
    n_fields_per_survey = ctx.n_fields_per_survey
    n_cosmo = ctx.n_theta_cosmo
    n_sims = 12
    n_shells = ctx.shell_n
    n_fields_per_survey = ctx.n_fields_per_survey
    n_fields_avail = n_cosmo*n_sims*n_shells
    n_fields_per_cosmo = n_shells*n_sims
    n_surveys_total = np.ceil(n_fields_per_cosmo/n_fields_per_survey).astype(int) * n_cosmo
    n_fields_total = n_surveys_total*n_fields_per_survey
    n_surveys_per_file = n_surveys_total//n_cosmo
    n_files = n_surveys_total//n_surveys_per_file
    batch_size = 334
    LOGGER.info(f'n_surveys_per_file={n_surveys_per_file} n_files={n_files}')
    
    n_reals_shuffled = 400
    n_reals_repeated = 100
    n_reals_fiducial = 1
    if test:
        n_reals_shuffled = 10
        n_reals_repeated = 2
        n_reals_fiducial = 1
        LOGGER.warning('===========> test!')

    n_surveys_per_index_shuffled = n_surveys_per_file*n_reals_shuffled
    n_surveys_per_index_repeated = n_surveys_per_file*n_reals_repeated
    n_surveys_per_index_fiducial = n_surveys_per_file*n_reals_fiducial
    n_batches_shuffled = n_surveys_per_index_shuffled//batch_size
    n_batches_repeated = n_surveys_per_index_repeated//batch_size
    n_batches_fiducial = n_surveys_per_index_fiducial//batch_size
    n_unique_astro_shuffled = 4000
    n_unique_astro_repeated = 100

    if test:
        n_batches = 10
        LOGGER.warning('===========> test!')

    
    def get_theta_astro(n_unique_astro, n_surveys_per_index, tag=''):

        # get astro parameters shuffled
        LOGGER.info(f'getting {tag} sequence with {n_unique_astro} unique astro points')
        theta_astro = ctx.theta_priors.sample(n_unique_astro, rule='sobol', rec=False, reset=True)
        theta_astro = perm_sequence(theta_astro, n_surveys_per_index)
        LOGGER.info(f'got {tag} theta astro with size={theta_astro.shape}')
        LOGGER.info(f'first={utils_arrays.arrstr(theta_astro[0])}') 
        LOGGER.info(f'last={utils_arrays.arrstr(theta_astro[-1])}')
        return theta_astro

    def get_dataset_astro(theta_astro):

        gen_astro = tf.data.Dataset.from_tensor_slices(theta_astro)
        gen_astro = gen_astro.batch(batch_size)
        gen_astro = gen_astro.repeat()
        gen_astro = iter(gen_astro)
        return gen_astro

    # get astro parameters repeated
    theta_astro_shuffled = get_theta_astro(n_unique_astro_shuffled, n_surveys_per_index_shuffled, tag='shuffled')
    theta_astro_repeated = get_theta_astro(n_unique_astro_repeated, n_surveys_per_index_repeated, tag='repeated')
    theta_astro_fiducial = np.repeat(utils_arrays.rec_to_arr(ctx.theta_priors.get_fiducial(), dtype=np.float32), batch_size, axis=0)
    gen_astro_shuffled = get_dataset_astro(theta_astro_shuffled)
    gen_astro_repeated = get_dataset_astro(theta_astro_repeated)
    gen_astro_fiducial = get_dataset_astro(theta_astro_fiducial)

    # load cosmo dataset
    from glob import glob
    files_cosmo = sorted(glob(f'{path_maps}/maps{index:04d}.tfrecords'))
    LOGGER.info(f'index={index} loaded {len(files_cosmo)} files')

    theta_transform_inverse = ctx.model_astro.get_func_theta_transform_inverse(method=ctx.training['transform_theta'])
    from deep_lss.apps.run_recordmaker import get_parse_inverse_func
    par_inv = get_parse_inverse_func(n_fields=n_fields_per_survey, n_pix=ctx.img_size, n_maps=3, n_zbins=4, n_y=2)

    gen_cosmo = tf.data.TFRecordDataset(files_cosmo, num_parallel_reads=1)
    gen_cosmo = gen_cosmo.map(par_inv, num_parallel_calls=1, deterministic=True)
    gen_cosmo = gen_cosmo.cache()
    gen_cosmo = gen_cosmo.repeat()
    gen_cosmo = gen_cosmo.batch(batch_size)
    gen_cosmo = iter(gen_cosmo)

    # get augmentations generator
    shuffles_len = int(1e6)
    np.random.seed(42)
    shuffles = np.array([np.random.permutation(n_fields_per_survey) for i in range(shuffles_len)])
    gen_shuff = tf.data.Dataset.from_tensor_slices(shuffles)
    gen_shuff = gen_shuff.batch(batch_size)
    gen_shuff = gen_shuff.repeat()
    gen_shuff = iter(gen_shuff)

    # process dataset 

    func_process_dataset = ctx.model_astro.get_fun_process_dataset(n_fields=ctx.n_fields_per_survey,
                                                                   n_pix=ctx.img_size,
                                                                   n_maps=ctx.n_maps,
                                                                   n_zbins=ctx.n_z_bins,
                                                                   batch_size=batch_size,
                                                                   transform_maps=ctx.training['transform_maps'])        
    # get number of galaxies per pixel depending on epoch
    def get_n_gal_per_pix(j, n_eff_target):

        n_gal_per_pix_target = n_eff_target * ctx.model_astro.pixel_area_arcmin2
        decay = lambda i, n0, half_time: n0*np.exp(-(np.log(2.)/half_time)*i)
        n_gal_per_pix = n_gal_per_pix_target + decay(j, ctx.training['neff_decay_n0'], ctx.training['neff_decay_half_time'])
        return tf.cast(n_gal_per_pix, dtype=tf.float32)

    from deep_lss.networks import layers
    layer_flip = layers.FlipFields(n_fields=n_fields_per_survey)

    @tf.function()
    def process(X, y_cosmo, y_astro, i_shuff, n_gal_per_pix):
        y = tf.concat((y_cosmo, y_astro), axis=1)
        X = func_process_dataset(X, y, n_gal_per_pix)
        X = layer_flip(tf.gather(X, i_shuff, axis=1, batch_dims=1))
        return X, y   

    def predictions(gen_cosmo, gen_astro, n_batches, tag='fiducial'):

        y_pred = {net.netname:[] for net in nets}
        y_true = {net.netname:[] for net in nets}
        
        LOGGER.info(f'running n_batches={n_batches} with batch_size={batch_size} for sequence={tag}')

        for i in LOGGER.progressbar(range(n_batches), at_level='info,debug', desc=f'getting predictions {tag} ... '):

            X, y_cosmo, i_cosmo = gen_cosmo.get_next()
            y_astro = gen_astro.get_next()
            i_shuff = gen_shuff.get_next()
            
            # get a single batch for all nets if there is no noise level augmentation
            if ctx.training['neff_decay_n0']==0:

                n_gal_per_pix_current = get_n_gal_per_pix(j=0, n_eff_target=ctx.model_astro.ng_eff) 
                X_, y_ = process(X, y_cosmo, y_astro, i_shuff, n_gal_per_pix_current)
                if tag=='fiducial':
                    select = np.nonzero(i_cosmo == ctx.theta_priors.fid_cosmo_id)[0]
                    X_ = tf.gather(X_, select)
                    y_ = tf.gather(y_, select)

            for net in nets:

                predictions = net.model(X_)
                yp = theta_transform_inverse(predictions[:,:len(net.output_select)], indices=net.output_select)
                yt = tf.gather(y_, net.output_select, axis=1)
                y_pred[net.netname].append(yp)
                y_true[net.netname].append(yt)

        concat_predictions(y_pred, y_true, tag=tag)

        return y_pred, y_true


    def concat_predictions(y_pred, y_true, tag=''):
                  
        for net in nets:
            y_pred[net.netname] = np.concatenate(y_pred[net.netname], axis=0)
            y_true[net.netname] = np.concatenate(y_true[net.netname], axis=0)
            net_theta = [ctx.theta_priors.theta_full[i] for i in net.output_select]
            y_pred[net.netname] = utils_arrays.arr_to_rec(y_pred[net.netname], net_theta, dtype=np.float32)
            y_true[net.netname] = utils_arrays.arr_to_rec(y_true[net.netname], net_theta, dtype=np.float32)
            LOGGER.info(f'{tag} net={net.netname} net_theta={net_theta} shape={y_pred[net.netname].shape}')

    def get_sensitivity_map():

        X, y_cosmo, i_cosmo = gen_cosmo.get_next()
        y_astro = gen_astro_fiducial.get_next()
        i_shuff = gen_shuff.get_next()

        nets_di_do = {}
        for net in nets:

            # assume noise level augmentaiton is converged to the final value for all nets
            select = np.nonzero(i_cosmo == ctx.theta_priors.fid_cosmo_id)[0]
            n_gal_per_pix_current = get_n_gal_per_pix(j=net.meta_info['step'], n_eff_target=ctx.model_astro.ng_eff) 
            
            X_noisy_, y_noisy_ = process(X, y_cosmo, y_astro, i_shuff, n_gal_per_pix_current)
            X_nonoise_, y_nonoise_ = process(X, y_cosmo, y_astro, i_shuff, n_gal_per_pix=1e7) # super large number of galaxies gives noise free data
            X_noisy_ = tf.gather(X_noisy_, select)
            y_noisy_ = tf.gather(y_noisy_, select)
            X_nonoise_ = tf.gather(X_nonoise_, select)
            y_nonoise_ = tf.gather(y_nonoise_, select)

            # this is just to make sure we have a tensor
            X_noisy_ = tf.constant(X_noisy_)
            with tf.GradientTape() as g:
                # add the input to the watch list
                g.watch(X_noisy_)
                # compute the output of the model
                o = net.model(X_noisy_)

            # batch_jacobian assumes that the gradiends of elements in the batch do not affect eachother
            # this is False, e.g. if you use batchnorm with training=True
            di_do = g.batch_jacobian(o, X_noisy_)

            X_noisy = net.network_imgs(X_noisy_)
            X_nonoise = net.network_imgs(X_nonoise_)
            nets_di_do[net.netname] = {'X_noisy': X_noisy.numpy(), 'X_nonoise': X_nonoise.numpy(), 'dX_do': di_do.numpy()}

            LOGGER.info(f'got sensitivity map for net {net.netname} with shape {di_do.shape}')

        return nets_di_do


    # with noise augmentation, processing for each net since they may be at different steps
    if ctx.training['neff_decay_n0']>0:
        raise Exception('noise augmentation currently not supported')

    if not sensitivity:

        # get predictions
        y_pred_shuffled, y_true_shuffled = predictions(gen_cosmo=gen_cosmo,
                                                       gen_astro=gen_astro_shuffled,  
                                                       n_batches=n_batches_shuffled, 
                                                       tag='shuffled')


        y_pred_repeated, y_true_repeated = predictions(gen_cosmo=gen_cosmo,
                                                       gen_astro=gen_astro_repeated,  
                                                       n_batches=n_batches_repeated, 
                                                       tag='repeated')

        y_pred_fiducial, y_true_fiducial = predictions(gen_cosmo=gen_cosmo,
                                                       gen_astro=gen_astro_fiducial,  
                                                       n_batches=n_batches_fiducial, 
                                                       tag='fiducial')




        return y_pred_shuffled, y_true_shuffled, y_pred_repeated, y_true_repeated, y_pred_fiducial, y_true_fiducial

    else:

        return get_sensitivity_map()

def get_example_data(gen_cosmo, gen_astro, func_process):

    # example images
    X, y_cosmo = gen_cosmo.get_next()
    y_astro = gen_astro.get_next()
    X_example, y_example = func_process(X, y_cosmo, y_astro)

    from deep_lss.networks.layers import PowerSpectra, Smoothing

    ps_layer = PowerSpectra(l_edges=np.linspace(100, 4000, 78+1), 
                            input_dims=[128, 128, 8], 
                            ang=5, 
                            n_ells=78)

    sigma = [0.315, 0.803, 1.681, 2.545]
    for i, sig in enumerate(sigma):
        blur = Smoothing(sigma=sig, n_channels=8)
        ps_example = ps_layer(blur(X_example))
        data_example = X_example, y_example, ps_example
        utils_io.store_example_images('.', data_example, tag=f'repeat_filter_sig{i}')
    import ipdb; ipdb.set_trace()
    return X_example, y_example, ps_example




def collect_prediction_results(indices, dir_output):
    """
    Merge results of predictions from job array into a single file.
    :param indices: indices to merge
    :param dir_output: dir where the output files are
    :return None: 
    """

    def add_net_y(n, y_pred, y_true):

        if n not in nets_y_pred.keys():
            nets_y_pred[n] = []
        if n not in nets_y_true.keys():
            nets_y_true[n] = []

        nets_y_pred[n] += [y_pred]
        nets_y_true[n] += [y_true]


    # iterate over two types of tests
    for seq in ['shuffled', 'repeated', 'fiducial']:

        # get storage containers
        nets_y_pred = {}
        nets_y_true = {}    

        # iterate over indices
        for index in LOGGER.progressbar(indices, at_level='info,debug'):
            
            # get index path            
            filepath_pred = get_filepath_predictions(dir_output, tag=f'{seq}_index{index:04d}')
                
            # check if file exists
            if not os.path.isfile(filepath_pred):
                LOGGER.error(f'file {filepath_pred} not found')
            
            else:

                # load file 
                with h5py.File(filepath_pred, 'r') as f:
                    for netname in f.keys():
                        add_net_y(netname, y_pred=np.array(f[f'{netname}/y_pred']), y_true=np.array(f[f'{netname}/y_true']))


        # merge results for test type
        for n, y_pred in nets_y_pred.items():
            nets_y_pred[n] = np.concatenate(y_pred)
            LOGGER.info(f'net={n} y_pred={len(nets_y_pred[n])}')

        for n, y_true in nets_y_true.items():
            nets_y_true[n] = np.concatenate(y_true)
            LOGGER.info(f'net={n} y_true={len(nets_y_true[n])}')

        # store output
        if len(nets_y_pred)>0:

            LOGGER.info(f'merged {len(indices)} indices for sequence={seq}')
            utils_io.store_predictions(dir_output=dir_output, 
                                       y_pred=nets_y_pred, 
                                       y_true=nets_y_true, 
                                       tag=seq)



