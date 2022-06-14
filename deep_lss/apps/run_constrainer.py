# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created on Nov 12 2020
author: Tomasz Kacprzak
"""

import os, warnings, argparse, h5py, numpy as np, time, itertools, importlib
from collections import OrderedDict
from deep_lss import utils_logging, utils_io, utils_arrays
from deep_lss.priors import ThetaPriors
from deep_lss.priors import convert_S8_to_sigma8, convert_sigma8_to_S8
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from deep_lss.configs import params as params_sims
from deep_lss.filenames import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

FIDUCIAL_COSMOLOGY_ID_SORTED = 34

SOBOL_CACHE = {}

def setup(args):

    description = 'Calculate cosmological constraints and validation tests'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--filename_config', type=str, required=True, 
                        help='configuration yaml file')
    parser.add_argument('--dir_output', type=str, required=True, 
                    help='output dir')
    parser.add_argument('--n_chains_per_index', default=1, type=int, 
                        help='how many chains to run per index')
    parser.add_argument('--models_filter', type=str, required=False, default=None,
                        help='only use these models')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    argk, _ = parser.parse_known_args(args)

    utils_logging.set_all_loggers_level(argk.verbosity)

    # get absolute paths
    argk.filename_config = utils_io.get_abs_path(argk.filename_config)
    argk.dir_output = utils_io.get_abs_path(argk.dir_output)
    dirpath_constraints = get_dirpath_constraints(argk.dir_output)
    utils_io.robust_makedirs(dirpath_constraints)

    return argk


def resources(args):

    reso = {}
    reso['main_memory'] = 2000
    reso['main_time_per_index'] = 1 # hours
    reso['main_nproc'] = 1
    reso['main_scratch'] = 2000
    reso['main_ngpu'] = 0
    reso['merge_memory'] = 16000
    reso['merge_time'] = 1
    reso['pass'] = {'partition':'hourly'}
    
    return reso


def main(indices, args):

    args = setup(args)
    ctx = utils_io.load_context(args.filename_config, models_filter=args.models_filter)
    dir_predictions = get_dirpath_predictions(args.dir_output)
    LOGGER.info(f'getting results from {dir_predictions}')

    for index in indices:


        chain_index = index//2
        
        if index % 2 == 0:

            mode = 'comparison'
            use_fiducial_cosmology=True
            sequence = 'fiducial'

        else:

            mode = 'validation'
            use_fiducial_cosmology=False
            sequence = 'repeated'

        LOGGER.info(f'============ running on index={index} chain_index={chain_index} mode={mode} use_fiducial_cosmology={use_fiducial_cosmology}')

        dict_out = run_constraints(ctx=ctx,
                                   file_pred=get_filepath_predictions(dir_predictions, tag=sequence), 
                                   dirpath_checkpoints=get_dirpath_checkpoints_likenet(args.dir_output),
                                   dir_out=get_dirpath_constraints(args.dir_output),
                                   chain_index=chain_index,
                                   fiducial_cosmology=use_fiducial_cosmology,
                                   test=args.test)

        filepath_out = get_filepath_chain_index(get_dirpath_constraints(args.dir_output), tag=mode, chain_index=chain_index)
        store_chain_index(filepath_out, dict_out)

        yield index



def merge(indices, args):

    args = setup(args)
    ctx = utils_io.load_context(args.filename_config, models_filter=args.models_filter)
    dirpath_constr = get_dirpath_constraints(args.dir_output)
    filepath_merged = get_filepath_chain_merged(dirpath_constr)
    filepath_stats = get_filepath_chain_stats(dirpath_constr)
    dir_predictions = get_dirpath_predictions(args.dir_output)

    n_chains_store = 20

    tags = ['comparison', 'validation']

    theta_names = ctx.theta_priors.theta_full

    n_chains = 0
    all_ok = True

    chain_stats = {'comparison': {n:{} for n in ctx.netnames},
                   'validation': {n:{} for n in ctx.netnames}}

    chain_stack = {}
    n_keep_per_chain = int(1e6)//(len(indices)//2)


    with h5py.File(filepath_merged, 'w') as f_out:

        for index in LOGGER.progressbar(indices, at_level='warning'):

            chain_index = index//2
            
            if index % 2 == 0:

                mode = 'comparison'
                use_fiducial_cosmology=True

            else:

                mode = 'validation'
                use_fiducial_cosmology=False

            filepath_in = get_filepath_chain_index(get_dirpath_constraints(args.dir_output), tag=mode, chain_index=chain_index)

            if not os.path.isfile(filepath_in):

                LOGGER.error(f'missing file {filepath_in}')
                all_ok = False

            else:

                with h5py.File(filepath_in, 'r') as f:


                    for n in ctx.netnames:

                        try:

                            samples = np.array(f[f'{n}/chain_{chain_index}/samples'])
                            y_true = np.array(f[f'{n}/chain_{chain_index}/y_true'])
                            y_pred = np.array(f[f'{n}/chain_{chain_index}/y_pred'])

                        except Exception as err:
                            LOGGER.error(f'failed to load {n} chain_index={chain_index} {err}')
                            pass

                        LOGGER.info(f'loaded {n} chain_index={chain_index}')

                        chain_stack.setdefault(n, [])
                                                
                        if chain_index < n_chains_store:

                            f_out[f'{mode}/{n}/samples_{chain_index:05d}'] = samples
                            f_out[f'{mode}/{n}/y_true_{chain_index:05d}']  = y_true
                            f_out[f'{mode}/{n}/y_pred_{chain_index:05d}']  = y_pred

                        chain_stats[mode][n][chain_index] = get_chain_stats(samples)

                        LOGGER.debug(f'{mode}: loaded {filepath_in} {n} {y_true}')

                    n_chains += 1

        LOGGER.info(f'merged n_chains={n_chains} wrote {filepath_merged}')
        dict_out = {'chain_stats':chain_stats}
        utils_io.write_to_pickle(filepath_stats, dict_out)

        
    # if all_ok:
    #     for index in indices:
    #         filepath_in = get_filepath_chain_index(args.dir_output, index)
    #         os.remove(filepath_in)

def add_S8(samples):
    
    samples = utils_arrays.add_cols(samples, names=['S8'])
    samples['S8'] =  samples['sigma_8']*(samples['Omega_m']/0.3)**(0.5)
    return samples

def get_chain_stats(samples):

    # add s8 to samples
    samples = add_S8(samples)

    if 'w' not in samples.dtype.names:
        samples = utils_arrays.add_cols(samples, names=['w'])
        samples['w'] = 1
        
    # init output
    stats = {}
    
    # get standard deviation of all params    
    cols = [c for c in samples.dtype.names if c!='log_prob']
    for c in cols:
        stats[f'std_{c}'] = np.std(samples[c])

    # get 68% interval
    cl68 = lambda x: np.percentile(x, q=84) - np.percentile(x, q=16)
    cols = [c for c in samples.dtype.names if c!='log_prob']
    for c in cols:
        stats[f'cl68_{c}'] = cl68(samples[c])


    # get contour area
    stats['area2d'] = get_figure_of_merit(s1=samples['Omega_m'], s2=samples['sigma_8'], w=samples['w'])
    LOGGER.debug('area2d: {:2.3f}'.format(stats['area2d']))

    return stats



def run_constraints(ctx, file_pred, dirpath_checkpoints, chain_index, dir_out, test=False, fiducial_cosmology=True):

    def get_cosmo_id(y_true):
        # this should have been added in predictor, but doing it here
        cosmo_id = np.zeros(len(y_true), dtype=np.int16)
        for i in range(len(ctx.theta_cosmo)):
            select = (y_true['Omega_m']==ctx.theta_cosmo[i]['Omega_m']) & (y_true['sigma_8']==ctx.theta_cosmo[i]['sigma_8'])
            cosmo_id[select] = i
        return cosmo_id
            
    def fix_first_to_mode(y):

        from scipy.stats import gaussian_kde
        from scipy.optimize import minimize
        yp_ = utils_arrays.rec_to_arr(y).T
        kde = gaussian_kde(yp_)
        ym_ = minimize(fun=lambda x: -kde.logpdf(x), x0=np.median(yp_, axis=1)).x
        LOGGER.debug(f'setting first sample to mode of all samples {ym_}')
        for i, c in enumerate(y.dtype.names):
            y[c][0] = ym_[i]
        return y

    # load data
    nets_y_true, nets_y_pred = utils_io.load_lssnet_predictions(file_pred)


    LOGGER.info(f'===========================> constraints for n_nets={len(nets_y_true.keys())}')

    # get model
    from deep_lss.networks.likemdn import LikeMDN

    # init output
    dict_out = {}

    # run a chain for each net
    for n, netname_current in enumerate(ctx.netnames):

        LOGGER.info(f'================> {n+1}/{len(ctx.netnames)} {netname_current}')

        # get input
        y_true = nets_y_true[netname_current]
        y_pred = nets_y_pred[netname_current]

        y_true_split, y_pred_split = utils_io.split_by_true_params(y_true, y_pred)
        
        # set first element to mode of all predictions
        y_pred_modes = np.concatenate([fix_first_to_mode(y)[[0]] for y in y_pred_split])
        y_pred = np.concatenate([y_pred_modes, y_pred])
        cosmo_id = get_cosmo_id(y_true)
        LOGGER.info(f'using only modes y_pred.shape={y_pred.shape} y_pred_modes.shape={y_pred_modes.shape}')
    
        # get net
        net = LikeMDN(nx=len(y_true.dtype), ny=len(y_pred.dtype), **ctx.likelihood_modelling['likenet'])
        file_likenet = get_checkpoint_name(dirpath_checkpoints, 'best',  tag=f'likemdn__{netname_current}')
        net.load(file_likenet)


        # how many samples?
        n_samples = ctx.analysis_settings['n_samples']
        if test:
            LOGGER.warning('==========> test!')
            n_samples = 3200

        if (chain_index == 0) and fiducial_cosmology:
            n_samples *= 10
            LOGGER.info(f'using the mode of summaries, running an extra-long chain with n_samples={n_samples}') 

        # run chains    
        dict_out[netname_current] = {}

        LOGGER.info(f'{chain_index} getting chains y_true={y_true[chain_index]}')

        try:
            post_samples = sample_posterior(net=net,
                                            yp=y_pred[chain_index],
                                            yt=y_true[chain_index],
                                            priors=ctx.theta_priors,
                                            n_samples=n_samples,
                                            method=ctx.analysis_settings['sampler'],
                                            test=test)

            dict_out[netname_current][chain_index] = dict(samples=post_samples, y_true=y_true[chain_index], y_pred=y_pred[chain_index], cosmo_id=cosmo_id[chain_index])

        except Exception as err:

            LOGGER.error(f'failed to get a chain {chain_index}, errmsg={err}')

    return dict_out


def store_chain_index(filepath, dict_out):

    with h5py.File(filepath, 'w') as f:
        for n in dict_out.keys():
            for k,v in dict_out[n].items():
                f.create_dataset(name=f'{n}/chain_{k}/samples', data=np.atleast_1d(v['samples']), compression='lzf', shuffle=True)
                f.create_dataset(name=f'{n}/chain_{k}/y_true', data=np.atleast_1d(v['y_true']))
                f.create_dataset(name=f'{n}/chain_{k}/y_pred', data=np.atleast_1d(v['y_pred']))
                f.create_dataset(name=f'{n}/chain_{k}/cosmo_id', data=np.atleast_1d(v['cosmo_id']))

    LOGGER.info(f'wrote {filepath} with {len(dict_out.keys())} items')


def sample_posterior(priors, method='emcee', test=False, **kwargs):

    if method=='emcee':

        samples = sample_posterior_emcee(priors=priors, test=test, **kwargs)

    elif method=='sobol':

        samples = sample_posterior_sobol(priors=priors, test=test, **kwargs, trim=1e-4)   

    return samples     


def sample_posterior_sobol(net, yp, yt, yt_full, priors, n_samples=None, test=False, trim=None):

    raise Exception('not implemented yet')

    # get sobol grid
    if 'grid' not in SOBOL_CACHE.keys():
        SOBOL_CACHE['grid'] = priors.load_theta_grid(rec=False)
    grid = SOBOL_CACHE['grid']
    n_grid = len(grid)

    n_dim = len(cols)


    # prep observation
    yp = np.atleast_2d(yp)

    # sequential calls
    LOGGER.info(f'sampling sobol grid with {n_grid} points')
    n_per_batch = 100000
    n_batches = int(np.ceil(n_grid/n_per_batch))
    log_prob = []
    for g in LOGGER.progressbar(np.array_split(grid, n_batches), at_level='debug', desc='getting sobol samples'):
        log_prob.append(net.log_like(g, yp))
    log_prob = np.concatenate(log_prob)

    # drop low probability region
    if trim is not None:
        select = np.isfinite(log_prob)
        grid = grid[select]
        log_prob = log_prob[select]
        sorting = np.argsort(log_prob)
        log_prob = log_prob[sorting]
        grid = grid[sorting,:]
        prob = expnorm(log_prob)
        cs = np.cumsum(prob)
        i = np.argmin(np.abs(cs-trim))
        grid = grid[i:,:]
        log_prob = log_prob[i:]
        LOGGER.info(f'returning {len(grid)} samples')
    
    samples = utils_arrays.arr_to_rec(samples, cols=cols, dtype=np.float32)
    samples = utils_arrays.add_cols(samples, ['log_prob'])
    samples['log_prob'] = log_prob

    return grid, log_prob



def sample_posterior_emcee(net, yp, yt, priors, n_samples, test=False):

    import emcee
    cols = [n for n in yp.dtype.names] # copy

    # define variables
    n_dim = len(cols)
    n_walkers = 256
    n_steps = int(n_samples/n_walkers)
    n_burnin_steps = 100
    # yp = np.atleast_2d(utils_arrays.rec_to_arr(convert_sigma8_to_S8(yp)))
    yp = np.atleast_2d(utils_arrays.rec_to_arr(yp))
    yt = np.atleast_2d(utils_arrays.rec_to_arr(yt))

    def log_posterior(y):

        theta = utils_arrays.arr_to_rec(y, cols)
        select = priors.in_prior(theta)
        # theta = convert_sigma8_to_S8(theta)
        y = utils_arrays.rec_to_arr(theta, cols)
        log_post = np.empty(len(y))
        log_post[~select] = -np.inf
        log_post[select] = net.log_like(y[select,:], yp)
        log_post[np.isnan(log_post)] = -np.inf
        return log_post

    # first scan to find good starting points

    # samples_sobol, logprob_sobol = sample_posterior_sobol(net, yp, yt_full, priors)
    # prob_sobol = expnorm(logprob_sobol)

    # p0_inds = np.argsort(logprob_sobol)[-n_walkers:]
    # p0 = samples_sobol[p0_inds]
    p0 = np.ones((n_walkers, n_dim))*yt
    p0 += np.random.normal(size=p0.shape)*1e-3
    
    # run mcmc
    sampler = emcee.EnsembleSampler(nwalkers=n_walkers, 
                                    ndim=n_dim, 
                                    log_prob_fn=log_posterior,
                                    vectorize=True)

    time_start = time.time()
    state = sampler.run_mcmc(p0, n_burnin_steps, progress=LOGGER.level in [utils_logging.logging.DEBUG])
    sampler.reset()
    sampler.run_mcmc(state, n_steps, progress=LOGGER.level in [utils_logging.logging.DEBUG, utils_logging.logging.INFO]);

    # get chain 
    samples = sampler.get_chain(flat=True)
    log_prob = sampler.get_log_prob(flat=True)
    LOGGER.info('got {} samples in {:2.2f} min'.format(len(samples), (time.time()-time_start)/60.))

    samples = utils_arrays.arr_to_rec(samples, cols=cols, dtype=np.float32)
    samples = utils_arrays.add_cols(samples, ['log_prob'])
    samples['log_prob'] = log_prob

    return samples



def expnorm(log_prob):

    prob = np.exp(log_prob - np.max(log_prob ))
    prob /= np.sum(prob)
    return prob


def get_figure_of_merit(s1, s2, w):
    

    lvl = 0.68
    bins1 = np.linspace(0, 0.6, 400)
    bins2 = np.linspace(0.4, 1.5, 400)
    bins_area = (bins1[-1]-bins1[0]) * (bins2[-1]-bins2[0])
    h = np.histogram2d(s1, s2, bins=(bins1, bins2), weights=w)[0]
    h /= h.sum()
    l = get_confidence_levels(h, levels=[lvl])[0]
    area = np.count_nonzero(h>l)/len(h.ravel()) * bins_area
    return area


def get_confidence_levels(de, levels, n_levels_check=1000, n_bins=50):

    from scipy.optimize import bisect

    levels_contour = []
    for cl in levels:
        cl_level = bisect(f=lambda l: np.sum(np.array(de > l) * de)-cl, a=0, b=1)
        levels_contour.append(cl_level)

    return levels_contour


