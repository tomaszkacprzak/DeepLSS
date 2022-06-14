import os, sys, shutil, stat, logging, subprocess, shlex, collections, datetime, numpy as np, pickle, importlib, h5py
from . import utils_logging, utils_arrays
from .filenames import *

LOGGER = utils_logging.get_logger(__file__)


def select_lssnets(ctx, models_filter=None):

    from fnmatch import fnmatch

    if models_filter is not None:

        lssnets_use = []
        if ',' in models_filter: # comma-spearated list of models

            for m in models_filter.split(','):
                for lssnet in ctx.lssnets:
                    if m == lssnet['name']:
                        lssnets_use.append(lssnet)

        else: # wildcard match

            for lssnet in ctx.lssnets:
                if fnmatch(lssnet['name'], models_filter):
                    lssnets_use.append(lssnet)
                else:
                    LOGGER.info(f"not using net {lssnet['name']}")

        assert len(lssnets_use)>0, f'no networks found matching filter={models_filter}'
        ctx.lssnets = lssnets_use

    # define networks
    ctx.netnames = []
    for i, lssnet in enumerate(ctx.lssnets):
        lssnet['module'] = importlib.import_module(f"deep_lss.networks.{lssnet['model']}")
        lssnet['init'] = getattr(lssnet['module'], lssnet['module'].class_name) 
        ctx.netnames.append(lssnet['name'])
        LOGGER.info(f"{i} using net {lssnet['name']}")
    ctx.n_lssnets = len(ctx.lssnets)
    LOGGER.warning(f"using {ctx.n_lssnets} nets")

def load_context(filename, models_filter=None, load_nets=True):

    from deep_lss.lss_astrophysics_model_batch import LssAstrophysicsModel
    from deep_lss.priors import ThetaPriors

    # read config
    conf = read_yaml(filename)
    for k,v in conf['files'].items():
        conf['files'][k] = v.replace('$SUBMIT_DIR', os.environ['SUBMIT_DIR'])

    # set defaults
    conf['analysis_settings'].setdefault('sampler', 'emcee')
    conf['analysis_settings'].setdefault('params_sim', 'params')
    conf['training'].setdefault('neff_decay_n0', 0)
    conf['training'].setdefault('neff_decay_half_time', 1)
    conf['training'].setdefault('transform_theta', 'unit')
    conf['training'].setdefault('transform_maps', True)
        
    LOGGER.info(f"loading params_sim={conf['analysis_settings']['params_sim']}")
    ctx = importlib.import_module(f"deep_lss.configs.{conf['analysis_settings']['params_sim']}")

    # write class attributes
    for key, val in conf.items():
        setattr(ctx, key, val)

    # full supported variables 
    ctx.probes_all = ['kg', 'dg']
    ctx.n_probes = len(ctx.probes_all)
    ctx.n_maps = 3
    ctx.n_z_bins = len(conf['analysis_settings']['redshift_bins'])
    ctx.z_max = 1.75

    # used variables 
    ctx.img_size = ctx.shell_pix_use
    ctx.n_channels = len(conf['analysis_settings']['redshift_bins']) * ctx.n_probes
    ctx.load_redshift_bins = np.sort(conf['analysis_settings']['redshift_bins'])
    ctx.theta_cosmo = load_theta_cosmo(conf['files']['sim_grid'], ctx)
    ctx.n_theta_cosmo = len(ctx.theta_cosmo)
    
    # filter nets to use    
    select_lssnets(ctx, models_filter=models_filter)



    # # init priors
    ctx.theta_priors = ThetaPriors(ctx=ctx)
    ctx.theta_all = ctx.theta_priors.theta_full
    ctx.n_theta_all = len(ctx.theta_all)
    ctx.n_theta = ctx.n_theta_all

    nz = []
    for i in ctx.load_redshift_bins:
        nz_file = ctx.lc_nz_files[i]
        nz_file_path = os.path.join(conf['files']['nz'], nz_file)
        nz_ = np.genfromtxt(nz_file_path)
        select = nz_[:,0]<ctx.z_max
        nz.append(nz_[select,:])
    nz = np.array(nz)
    LOGGER.info(f'loaded nz grids, shape={nz.shape}')

    
    # # initialize astrophysics model
    ctx.model_astro = LssAstrophysicsModel(params_astro_const=conf['params_astro_const'], 
                                           theta_priors=ctx.theta_priors, 
                                           probes=conf['analysis_settings']["probes"],
                                           nz=nz,
                                           img_size=ctx.img_size)

    bs, be = min(ctx.load_redshift_bins), max(ctx.load_redshift_bins)+1

    import tensorflow as tf
    def get_func_select_redshift_bins(bs, be):

        bsc = tf.constant(bs)
        bec = tf.constant(be)

        @tf.function()
        def select_redshift_bins(X, y, c):

            return X[...,bsc:bec], y, c

        return select_redshift_bins
        
    ctx.select_redshift_bins = get_func_select_redshift_bins(bs, be)


    return ctx




def get_networks(ctx, dirpath_checkpoints=None, dirpath_logs=None, strategy=None, epoch='last'):
    
    import tensorflow as tf

    def get_probe_channel_slicing(probes):

        if probes == 'cp':
            probe_slicing = [0, ctx.n_probes*ctx.n_z_bins]
        elif probes == 'kg':
            probe_slicing = [0*ctx.n_z_bins, 1*ctx.n_z_bins] # first in the stack
        elif probes == 'dg':
            probe_slicing = [1*ctx.n_z_bins, 2*ctx.n_z_bins] # second in the stack
        return probe_slicing

    def get_probe_theta_gathering(probes, case, priors):

        theta_map = {p:i for i, p in enumerate(priors.theta_full)}

        if case in ['baseline']:

            if probes == 'cp':
                probe_params = ['Omega_m', 'sigma_8', 'A_IA', 'b_gal', 'eta_A_IA', 'eta_b_gal']

            elif probes == 'kg':
                probe_params = ['Omega_m', 'sigma_8', 'A_IA', 'eta_A_IA']

            elif probes == 'dg':
                probe_params = ['Omega_m', 'sigma_8', 'b_gal', 'eta_b_gal']

        elif case in ['stochasticity', 'no_clustering_for_kg']:
    
            if probes == 'cp':
                probe_params = ['Omega_m', 'sigma_8', 'A_IA', 'b_gal', 'r_stoch', 'eta_A_IA', 'eta_b_gal']

            elif probes == 'kg':
                probe_params = ['Omega_m', 'sigma_8', 'A_IA', 'eta_A_IA']

            elif probes == 'dg':
                probe_params = ['Omega_m', 'sigma_8', 'b_gal', 'r_stoch', 'eta_b_gal']

        elif case in ['nonlin_bias']:
    
            if probes == 'cp':
                probe_params = ['Omega_m', 'sigma_8', 'A_IA', 'b_gal', 'r_stoch', 'eta_A_IA', 'eta_b_gal', 'b2_gal', 'eta_b2_gal']

            elif probes == 'kg':
                probe_params = ['Omega_m', 'sigma_8', 'A_IA', 'eta_A_IA']

            elif probes == 'dg':
                probe_params = ['Omega_m', 'sigma_8', 'b_gal', 'r_stoch', 'eta_b_gal', 'b2_gal', 'eta_b2_gal']
    
        elif case == 'super-poissonian':

            raise Exception('super-poissonian not implemented yet')

        gather_theta = [theta_map[p] for p in probe_params if priors.param_variable[p]]
        return gather_theta

    LOGGER.info(f"using galaxy counts noise model {ctx.params_astro_const['galaxy_counts_noise_model']}")
    nets = []
    for m, lssnet in enumerate(ctx.lssnets):

        with tf.name_scope(f"lssnet_{lssnet['name']}") as scope:

            smoothing_sigma_pix = np.array(ctx.lssnets[m]['smoothing_sigma_arcmin'])/ctx.model_astro.pixel_size_arcmin
            probe_channel_select = get_probe_channel_slicing(ctx.lssnets[m]['probes'])
            probe_output_select = get_probe_theta_gathering(ctx.lssnets[m]['probes'], ctx.params_astro_const['galaxy_counts_noise_model'], ctx.theta_priors)

            LOGGER.info(f"==================== creating lssnet {m+1}/{ctx.n_lssnets} {lssnet['name']}")
            LOGGER.info(f"------ probes={ctx.lssnets[m]['probes']} probe_output_select={probe_output_select} probe_channel_select={probe_channel_select}")
            LOGGER.info(f"------ smoothing_sigma={ctx.lssnets[m]['smoothing_sigma_arcmin']} arcmin / {smoothing_sigma_pix} pix")


            net = lssnet['init'](n_output=ctx.n_theta, 
                                 img_size=ctx.img_size,
                                 smoothing_sigma=smoothing_sigma_pix,
                                 n_channels_all=ctx.n_channels,
                                 channels_select=probe_channel_select,
                                 output_select=probe_output_select,
                                 strategy=strategy,
                                 mosaic=ctx.n_fields_per_survey,
                                 netname=lssnet['name'],
                                 angle=ctx.params_astro_const['map_patch_size_deg'],
                                 **lssnet)

            if dirpath_checkpoints is not None:

                try:

                    filepath_check = get_checkpoint_name(dirpath_checkpoints, epoch=epoch, tag=lssnet['name'])
                    net.load(filepath_check)

                except Exception as err:

                    LOGGER.error(f'failed to load {filepath_check} err={err}, trying previous...')
                    filepath_check = get_checkpoint_name(dirpath_checkpoints, epoch=f'{epoch}_prev', tag=lssnet['name'])
                    net.load(filepath_check)

            if dirpath_logs is not None:

                dirpath_net_log = os.path.join(dirpath_logs, lssnet['name'])
                if not os.path.isdir(dirpath_net_log):
                    os.makedirs(dirpath_net_log)
                    LOGGER.info(f'created dir {dirpath_net_log}')
                net.writer = tf.summary.create_file_writer(dirpath_net_log)

            LOGGER.info(f"------ downsampling={net.nn_config['downsample']} orig_dim={net.nn_config['orig_dim']}")
            LOGGER.info(f"------ model={lssnet['model']} n_params={net.model.count_params()}")
            net.summary()
            nets.append(net)

    return nets


def load_lssnet_predictions(file_pred, squeeze=False, flatten=False, models_filter=None):

    import h5py

    nets_y_pred = {}
    nets_y_true = {}

    with h5py.File(file_pred, 'r') as f:
        for netname in f.keys():
            nets_y_pred[netname] = np.array(f[netname]['y_pred'])
            nets_y_true[netname] = np.array(f[netname]['y_true'])

    LOGGER.info(f'loaded {file_pred} with nets={nets_y_true.keys()}, shape {nets_y_pred[netname].shape}')

    if flatten:
        for n in nets_y_pred.keys():
            nets_y_pred[n] = nets_y_pred[n].flatten()
        for n in nets_y_true.keys():
            nets_y_true[n] = nets_y_true[n].flatten()

    if squeeze:
        for n in nets_y_true.keys():
            nets_y_true[n] = nets_y_true[n].squeeze()
        for n in nets_y_true.keys():
            nets_y_pred[n] = nets_y_pred[n].squeeze()
    
    return nets_y_true, nets_y_pred


def split_by_true_params(y_true, y_pred, y_samp=None):
        
    hashing = utils_arrays.rec_to_arr(y_true).sum(axis=1)
    uv = np.unique(hashing)
    y_true_split, y_samp_split, y_pred_split = [], [], []
    for v in uv:
        select = hashing == v
        y_true_split.append(y_true[select])
        y_pred_split.append(y_pred[select])       
        if y_samp is not None:
            y_samp_split.append(y_samp[select])
    
    if y_samp is not None: 
        return y_true_split, y_pred_split, y_samp_split
    else:
        return y_true_split, y_pred_split
    

def load_theta_cosmo(dirpath, params):

    # # OmegaM OmegaL w0    wa     Omegab hubble sigma8 nspec
    names = ['Omega_m', 'Omega_Lambda', 'w0', 'wa', 'Omega_b', 'H0', 'sigma_8', 'n_spec']
    dtype = np.dtype([(n, np.float64) for n in names])
    theta_cosmo = np.genfromtxt(os.path.join(dirpath, params.cosmo), dtype=dtype)
    return theta_cosmo

def copy_dataset_to_local_scratch(args, ctx, sequence='tomo', n_files=None):

    path_cosmo = get_dirpath_tfrecords(dirpath_maps=ctx.files['projected_maps'], npix=ctx.shell_pix_use, seq=sequence)
    path_astro = ctx.files['astro_prior_grid'] 

    if not args.copy_dataset:

         return path_cosmo, path_astro

    else:    

        import random
        
        dir_scratch_cosmo = '/scratch/dataset_{:x}/'.format(random.getrandbits(32))
        if not os.path.isdir(dir_scratch_cosmo):
            os.makedirs(dir_scratch_cosmo)
            LOGGER.info(f'created dir {dir_scratch_cosmo}')
        LOGGER.info(f'copying {path_cosmo} -> {dir_scratch_cosmo}')

        files_copy = sorted(os.listdir(path_cosmo))[:n_files]
        for f in LOGGER.progressbar(files_copy, at_level='info,debug'):
            shutil.copyfile(src=os.path.join(path_cosmo, f), 
                            dst=os.path.join(dir_scratch_cosmo, f))
        LOGGER.info('done copying cosmo')


        dir_scratch_astro = '/scratch/dataset_{:x}/'.format(random.getrandbits(32))
        if not os.path.isdir(dir_scratch_astro):
            os.makedirs(dir_scratch_astro)
            LOGGER.info(f'created dir {dir_scratch_astro}')
        LOGGER.info(f'copying {path_astro} -> {dir_scratch_astro}')

        files_copy = sorted(os.listdir(path_astro))
        for f in LOGGER.progressbar(files_copy, at_level='info,debug'):
            shutil.copyfile(src=os.path.join(path_astro, f), 
                            dst=os.path.join(dir_scratch_astro, f))
        LOGGER.info('done copying astro')

        return dir_scratch_cosmo, dir_scratch_astro



def store_predictions(dir_output, y_pred, y_true, y_samp=None, tag='test'):

    filepath_pred = get_filepath_predictions(dir_output, tag)
    with h5py.File(filepath_pred, 'w') as f:

        for i, k in enumerate(y_pred.keys()):

            f.create_dataset(name=f'{k}/y_true', data=y_true[k], compression='lzf', shuffle=True)
            f.create_dataset(name=f'{k}/y_pred', data=y_pred[k], compression='lzf', shuffle=True)

            # if needed, store predictions from likenet
            if y_samp is not None:
                f.create_dataset(name=f'{k}/y_samp', data=y_samp[k], compression='lzf', shuffle=True)

    LOGGER.info(f'wrote {filepath_pred}')
    return filepath_pred



def test_tensorflow():

    import tensorflow as tf
    LOGGER.info('host {}'.format(os.environ['HOSTNAME']))
    LOGGER.info("All devices: {}".format(tf.config.list_physical_devices()))
    LOGGER.info("Num GPUs Available: {}".format(len(tf.config.list_physical_devices('GPU'))))

def store_example_images(dir_output, data_example, tag=''):
    
    file_out=get_filepath_example_images(dir_output, tag=tag)
    with h5py.File(file_out, 'w') as f:
        for k, v in data_example.items():
            f.create_dataset(name=k, data=v, compression='lzf', shuffle=True)
    LOGGER.info(f'wrote {file_out}')


def write_to_pickle(filepath, obj, compression='none', compression_kwargs={}):

    compression_kwargs.setdefault('compresslevel', 4)
    compression_types =  ['none', 'lzf', 'bz2', 'gzip']

    if compression.lower()=='none':
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    elif compression.lower()=='lzf':
        import lzf
        with lzf.open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    elif compression.lower()=='bz2':
        import bz2
        with bz2.open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    elif compression.lower()=='gzip':
        import gzip
        with gzip.open(filepath, 'wb', **compression_kwargs) as f:
            pickle.dump(obj, f)

    else: 
        raise Exception('uknown compression {}, use {}'.format(compression, str(formats)))
    LOGGER.info(f'wrote {filepath}')


def read_from_pickle(filepath, compression='none'):

    compression_types =  ['none', 'lzf', 'bz2', 'gzip']

    if compression.lower()=='none':
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
    elif compression.lower()=='lzf':
        import lzf
        with lzf.open(filepath, 'rb') as f:
            obj = pickle.load(f)
    elif compression.lower()=='bz2':
        import bz2
        with bz2.open(filepath, 'rb') as f:
            obj = pickle.load(f)
    elif compression.lower()=='gzip':
        import gzip
        with gzip.open(filepath, 'rb') as f:
            obj = pickle.load(f)
    else: 
        raise Exception('uknown compression {}, use {}'.format(compression, str(formats)))

    LOGGER.debug(f'read {filepath}')
    return obj

def read_yaml(filename):

    import yaml
    with open(filename, 'r') as fobj:
        d = yaml.load(fobj, Loader=yaml.FullLoader)
    LOGGER.debug('read yaml {}'.format(filename))
    return d

def write_yaml(filename, d):
        
    import yaml

    with open(filename, 'w') as f:
        stream = yaml.dump(d, default_flow_style=False, width=float("inf"))
        f.write(stream.replace('\n- ', '\n\n- '))

    LOGGER.debug('wrote yaml {}'.format(filename))


def get_abs_path(path):

    if '@' in path and ':/' in path:
        abs_path = path

    elif os.path.isabs(path):
        abs_path = path

    else:
        if 'SUBMIT_DIR' in os.environ:
            parent = os.environ['SUBMIT_DIR']
        else:
            parent = os.getcwd()

        abs_path = os.path.join(parent, path)

    return abs_path

def robust_makedirs(path):

    if is_remote(path):
        LOGGER.info('Creating remote directory {}'.format(path))
        host, path = path.split(':')
        cmd = 'ssh {} "mkdir -p {}"'.format(host, path)
        subprocess.call(shlex.split(cmd))

    elif not os.path.isdir(path):
        try:
            os.makedirs(path)
            LOGGER.info('Created directory {}'.format(path))
        except FileExistsError as err:
            LOGGER.error(f'already exists {path}')

def robust_remove(dirpath):

    if os.path.isdir(dirpath):
        LOGGER.info(f'removing {dirpath}')
        shutil.rmtree(dirpath)
    else:
        LOGGER.error(f'dir {dirpath} not found')


def ensure_permissions(path, verb=True):
    val = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    os.chmod(path, val)
    if verb:
        LOGGER.debug('Changed permissions for {} to {}'.format(path, oct(val)))


def access_remove(filepath, verb=True):
    cmd = 'chmod 000 {:s}'.format(filepath)
    if verb:
        LOGGER.info('changing permissions of file {:s} to 000'.format(filepath))
    os.system(cmd)


def access_grant(filepath, verb=True):
    cmd = 'chmod 755 {:s}'.format(filepath)
    if verb:
        LOGGER.info('changing permissions of file {:s} to 755'.format(filepath))
    os.system(cmd)

def is_remote(path):
    return '@' in path and ':/' in path


def save_hdf(filename, arr, verb=True, **kwargs):

    import h5py

    # kwargs.setdefault('compression', 'lzf')
    # kwargs.setdefault('shuffle', True)
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset(name='data', data=arr, **kwargs)

    if verb:
        LOGGER.info('saved {}'.format(filename))


def load_hdf(filename, first_row=-1, last_row=-1, verb=True, field='data'):

    import h5py

    with h5py.File(name=filename, mode='r')  as f:

        if (first_row > -1) & (last_row > -1) & (first_row < last_row):
            if first_row < last_row:
                data = np.array(f[field][first_row:last_row])
            else:
                raise Exception("field={}: first_row={} should be smaller than last_row={}".format(field, first_row, last_row))
        else:
            data = np.array(f[field])

    if verb:
        LOGGER.info('loaded {}'.format(filename))

    return data