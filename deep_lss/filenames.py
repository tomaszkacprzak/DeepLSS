import os

def get_dirpath_checkpoints(dir_output):

    return os.path.join(dir_output, f'checkpoints_lssnet')

def get_checkpoint_name(dirpath_checkpoints, epoch, tag='cnn'):

    return os.path.join(dirpath_checkpoints, f'checkpoint__{tag}__epoch{epoch}')

def get_dirpath_predictions(dir_output):

    return os.path.join(dir_output, 'predictions')

def get_filepath_predictions(dirpath, tag):

    return os.path.join(dirpath, f'predictions__{tag}.h5')

def get_filepath_example_images(dirpath, tag=''):

    return os.path.join(dirpath, f'example_images_{tag}.h5')

def get_dirpath_lssnet_logs(dir_output):

    return os.path.join(dir_output, f'lssnet_logs')

def get_dirpath_likenet_logs(dir_output):

    return os.path.join(dir_output, 'likenet_logs')

def get_filepath_chain_index(dirpath, tag, chain_index):

    return os.path.join(dirpath, f'constraints_{tag}_chain{chain_index:04d}.h5')

def get_filepath_chain_merged(dirpath):

    return os.path.join(dirpath, f'constraints_merged.h5')

def get_filepath_chain_stats(dirpath):

    return os.path.join(dirpath, f'constraints_stats.pkl')

def get_filepath_sensitivity_maps(dirpath, tag):

    return os.path.join(dirpath, f'sensitivity_maps_{tag}.pkl')

def get_filepath_chain_merged_small(dirpath):

    return os.path.join(dirpath, f'constraints_merged_small.h5')

def get_dirpath_constraints(dirpath):

    return os.path.join(dirpath, 'constraints')    

def get_dirpath_checkpoints_likenet(dir_output):

    return os.path.join(dir_output, f'checkpoints_likenet')

def get_dirpath_weights(dirpath_output, cosmo_str):

    dirpath_out = os.path.join(dirpath_output, cosmo_str)
    filepath_out = os.path.join(dirpath_out, 'maps', "n_of_z_weights.h5")
    return filepath_out

def get_filepath_kappa_maps(dirpath_output, cosmo_str):

    dirpath_out = os.path.join(dirpath_output, 'maps', cosmo_str)
    return os.path.join(dirpath_out, "kappa_maps.h5")

def get_dirpath_tfrecords(dirpath_maps, seq, npix=128):

    dirpath_tfrecords = os.path.join(dirpath_maps, f'tfrecords_{seq}_npix{npix}')
    return dirpath_tfrecords

def get_filename_tfrecords_map(i):

    return f'maps{i:04d}.tfrecords'