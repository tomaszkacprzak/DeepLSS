# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created on Nov 12 2020
author: Tomasz Kacprzak
"""

import os, warnings, argparse, h5py, numpy as np, time, logging, itertools
from collections import OrderedDict
from deep_lss import utils_logging, utils_io, utils_weights_calc
from deep_lss.utils_weights_calc import cosmo_rename
from deep_lss.filenames import *

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
    parser.add_argument('--skip_weights', action='store_true',
                        help='do not compute projection weights')
    argk, _ = parser.parse_known_args(args)

    utils_logging.set_all_loggers_level(argk.verbosity)

    # get absolute paths
    argk.filename_config = utils_io.get_abs_path(argk.filename_config)
    argk.dirpath_output = utils_io.get_abs_path(argk.dirpath_output)


    # make output directories
    utils_io.robust_makedirs(argk.dirpath_output)
    utils_io.robust_makedirs(os.path.join(argk.dirpath_output, 'maps'))

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
    conf = load_config(args.filename_config)
    ctx = utils_io.load_context(args.filename_config)


    if args.test:
        LOGGER.critical('========= test mode!')

    for index in indices:

        LOGGER.error(f'running on index {index}')

        if not args.skip_weights:

            get_weights(ctx=ctx,
                        dirpath_output=args.dirpath_output,
                        base_dir=conf['files']['sim_grid'], 
                        i_cosmo=index,
                        dirpath_nz=conf['files']['nz'], 
                        test=args.test)

        project_cone(ctx=ctx,
                     dirpath_output=args.dirpath_output,
                     base_dir=conf['files']['sim_grid'], 
                     i_cosmo=index,
                     test=args.test)

        yield index

def missing(indices, args):

    from deep_lss.utils_weights_calc import cosmo_rename

    args = setup(args)
    conf = load_config(args.filename_config)
    ctx = utils_io.load_context(args.filename_config)
    cosmos = np.genfromtxt(os.path.join(conf['files']['sim_grid'], ctx.cosmo))

    list_missing = []
    for index in indices:
        for j in range(ctx.n_per_cosmo):
            cosmo = cosmos[index]
            cosmo_str, _ = cosmo_rename(cosmo, j, ctx)
            filepath_out = get_filepath_kappa_maps(args.dirpath_output, cosmo_str)
            if not os.path.isfile(filepath_out):
                LOGGER.error(f'missing file cosmo={index} j={j} {filepath_out}')
                list_missing.append(index)

    list_missing = list(set(list_missing))
    LOGGER.info(f'missing {len(list_missing)}, {list_missing}')
    return list_missing


def load_config(filepath_config):

    if filepath_config.endswith('.pkl'):
        conf = utils_io.read_from_pickle(filepath_config)
    else:
        conf = utils_io.read_yaml(filepath_config)
    return conf

def get_shell_files(file_path):

    shell_files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f)) and "shells_num_1000" in f]
    LOGGER.info(f'found {len(shell_files)} shell files')
    return np.sort(shell_files)


def get_weights(ctx, base_dir, i_cosmo, dirpath_output, dirpath_nz, test=False):
    """
    Calculated the weights of the mean redshift dists of a given cosmology
    :param base_dir: Path where all the simulations are
    :param i_cosmo: Which cosmo to take from the cosmo file
    :return: Saves array shape (2, n_dist, n_shells) first dim is signal / IA, second the redshift distributions
             specified in the params, third dim contains the weights for each shell (sorted)
    """

    # from deep_lss import utils_weights_calc import cosmo_rename, shell_redshift, dimless_com_nz, w_nz, w_IA, w_dg
    # from deep_lss import utils_weights_calc import calculate_pixarea, scale_kappa_weight, scale_IA_weight, scale_dg_weight
    from deep_lss import utils_weights_calc as utils_w

    # read out cosmology
    cosmos = np.genfromtxt(os.path.join(base_dir, ctx.cosmo))
    cosmo = cosmos[i_cosmo]

    # calculate the weights
    H0 = cosmo[5]*100
    OmegaM = cosmo[0]
    OmegaL = cosmo[1]

    # Get cosmo str
    cosmo_str, _ = utils_w.cosmo_rename(cosmo, 0, ctx)
    LOGGER.info(f"Getting shells from cosmo (num={i_cosmo}): {cosmo_str}")

    # get path to shells
    file_path = os.path.join(base_dir, cosmo_str, "shells")

    # get shell files for redshift
    shell_files = get_shell_files(file_path)

    zbin_weights_kg = []
    zbin_weights_ia = []
    zbin_weights_dg = []
    shells_z = np.zeros([len(shell_files),2])
    
    last_shell = 2 if test else None


    for nz_file in ctx.lc_nz_files:
        LOGGER.info(f"Starting with nz file: {nz_file}")
        weights_kg = []
        weights_ia = []
        weights_dg = []

        # Get the redshift distribution
        nz, z_points = load_nz_interp(dirpath_nz, nz_file)

        for i, f in LOGGER.progressbar(list(enumerate(shell_files[:last_shell])), at_level='info'):
            LOGGER.debug(f"Dealing with shell: {f} {i}/{len(shell_files)}")

            # Get redshift and delta_z
            z, delta_z = utils_w.shell_redshift(f, ctx)
            shells_z[i] = z, delta_z

            # Calculate Pixelarea (assume all pixels have same area)
            pix_area = utils_w.calculate_pixarea(ctx, z, delta_z, H0, OmegaL, OmegaM)

            # Calculate Factors
            wkg = utils_w.w_nz(z, nz, ctx.lc_z_bounds, delta_z, z_points, H0, OmegaL, OmegaM)
            wia = utils_w.w_IA(z, nz, ctx.lc_z_bounds, delta_z, z_points, H0, OmegaL, OmegaM)
            wdg = utils_w.w_dg(z, nz, ctx.lc_z_bounds, delta_z, z_points, H0, OmegaL, OmegaM)

            # append redshift weights
            weights_kg.append(utils_w.scale_kappa_weight(wkg, H0, OmegaM, pix_area))
            weights_ia.append(utils_w.scale_IA_weight(wia, pix_area))
            weights_dg.append(utils_w.scale_dg_weight(wdg, pix_area))

        # append redshift weights
        zbin_weights_kg.append(np.asarray(weights_kg))
        zbin_weights_ia.append(np.asarray(weights_ia))
        zbin_weights_dg.append(np.asarray(weights_dg))

    # collect everything
    weights = np.array([zbin_weights_kg, zbin_weights_ia, zbin_weights_dg])
    LOGGER.info(f'created weights array, shape={weights.shape}')

    # save the weights
    LOGGER.info(f"Saving weights to {dirpath_output}")

    # Save weights
    for num in range(ctx.n_per_cosmo):
        cosmo_str, _ = utils_w.cosmo_rename(cosmo, num, ctx)
        filepath_w = get_dirpath_weights(dirpath_output, cosmo_str)
        save_weigths(filepath_w, weights, shells_z)


def project_cone(ctx, base_dir, i_cosmo, dirpath_output, test=False):
    """
    Projects the shells of a given cosmology and saves them to disk
    :param base_dir: Path where all the simulations are
    :param i_cosmo: Which cosmo to take from the cosmo file
    """

    from deep_lss.utils_weights_calc import cosmo_rename

    # indices of kappa maps:
    # kappa_maps.shape == (3, 5, 1000, 128, 128)
    # kappa_maps[0] = convergence
    # kappa_maps[1] = intrinsic alignment
    # kappa_maps[2] = galaxy density

    # read out cosmology
    cosmos = np.genfromtxt(os.path.join(base_dir, ctx.cosmo))
    cosmo = cosmos[i_cosmo]

    # test
    last_shell = 2 if test else None

    for j in range(ctx.n_per_cosmo):
        # Get cosmo str
        cosmo_str, _ = cosmo_rename(cosmo, j, ctx)
        LOGGER.info(f"Getting shells from cosmo (num={i_cosmo}): {cosmo_str}")

        # get path to shells
        file_path = os.path.join(base_dir, cosmo_str, "shells")

        # get shell files for redshift
        shell_files = get_shell_files(file_path)

        # load the weights
        filepath_w = get_dirpath_weights(dirpath_output, cosmo_str)
        weights = utils_io.load_hdf(filepath_w, field='weights')
        LOGGER.debug(f'read {filepath_w} shape={weights.shape}')

        # project
        for i_shell, shell in LOGGER.progressbar(list(enumerate(shell_files[:last_shell])), at_level='info'):
            # load the parts
            LOGGER.debug(f"Loading shells: {shell}")
            shell_parts = np.load(os.path.join(file_path, shell))

            # init kappa maps
            if i_shell == 0:
                kappa_maps = np.zeros(weights.shape[:-1] + shell_parts.shape)

            # add
            kappa_maps += weights[...,i_shell,None,None,None]*shell_parts[None,None,...]

        # convert to the right type/shape
        kappa_maps = kappa_maps.astype(np.float32)
        kappa_maps = np.moveaxis(kappa_maps, 0, -1)
        kappa_maps = np.moveaxis(kappa_maps, 0, -1)

        # save
        filepath_out = get_filepath_kappa_maps(dirpath_output, cosmo_str)
        save_kappa_maps(filepath_out, kappa_maps)

def load_nz_interp(dirpath_nz, nz_file):

    from scipy.interpolate import interp1d
    nz_file_path = os.path.join(dirpath_nz, nz_file)
    nz_tabu = np.genfromtxt(nz_file_path)
    z_points = nz_tabu[:, 0]
    nz = interp1d(nz_tabu[:, 0], nz_tabu[:, 1], bounds_error=False, fill_value=0.0, kind="cubic")
    return nz, z_points

def save_kappa_maps(filepath_out, kappa_maps):
    
    utils_io.robust_makedirs(os.path.dirname(filepath_out))
    with h5py.File(filepath_out, 'w') as f: 
        for i in range(kappa_maps.shape[0]):
            f.create_dataset(name=str(i), data=kappa_maps[i], chunks=None)
    LOGGER.info(f'stored {filepath_out} with {kappa_maps.shape[0]} datasets')

def save_weigths(filepath_w, weights, shells_z):

    utils_io.robust_makedirs(os.path.dirname(filepath_w))
    with h5py.File(filepath_w, 'w') as f:
        f.create_dataset(name='weights', data=weights)
        f.create_dataset(name='shell_z', data=shells_z)
    LOGGER.info(f'wrote {filepath_w}')



def perm_sequence(elements, length):

    perms = [np.random.permutation(elements) for _ in range(int(np.ceil(length/len(elements))))]    
    return np.concatenate(perms)[:length]
