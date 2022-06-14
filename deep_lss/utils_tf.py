from . import utils_logging
LOGGER = utils_logging.get_logger(__file__)

# may put this function in another utility file
def import_tensorflow():
    # Filter tensorflow version warnings
    import os
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    return tf

def test_tensorflow():

    import tensorflow as tf
    LOGGER.info('host {}'.format(os.environ['HOSTNAME']))
    LOGGER.info("All devices: {}".format(tf.config.list_physical_devices()))
    LOGGER.info("Num GPUs Available: {}".format(len(tf.config.list_physical_devices('GPU'))))
