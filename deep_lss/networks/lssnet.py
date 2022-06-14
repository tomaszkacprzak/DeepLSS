import tensorflow as tf
from deep_lss.networks import layers
from deep_lss.networks import losses
from deep_lss.networks.autoclip_tf import AutoClipper
from deep_lss import utils_logging, utils_io
LOGGER = utils_logging.get_logger(__file__)


class LssNET():

    def __init__(self, n_output, img_size, mosaic=1, file_load=None, smoothing_sigma=1, netname='lssnet', learning_rate=0.00005, grad_clip_percentile=None, channels_select=[0,None], n_channels_all=8, output_select=None, **kwargs):
        """
        :param smoothing_sigma: Apply Gaussian smoothing with sigma (pixels)
        """

        self.n_output = n_output
        self.mosaic = mosaic
        self.mosaic_side = tf.cast(tf.math.sqrt(tf.cast(mosaic, tf.float32)), tf.int64)
        self.img_size = img_size
        self.smoothing_sigma = tf.constant(smoothing_sigma, dtype=tf.float32)
        self.meta_info = {'step':0, 'epoch':0}
        self.netname = netname
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, 
                                            gradient_transformers=None if grad_clip_percentile is None else [AutoClipper(grad_clip_percentile)], 
                                            name=f"opt_{self.netname}",) 
        self.grad_clip_percentile = grad_clip_percentile
        self.checkpoint = None
        self.n_channels_all = n_channels_all
        self.channels_select_start = channels_select[0]
        self.channels_select_end = channels_select[1]
        self.all_channels = ((self.channels_select_start==0) and (self.channels_select_end==self.n_channels_all))
        self.n_channels = len(range(n_channels_all)[channels_select[0]:channels_select[1]])
        if self.mosaic > 1:
            self.input_shape = (self.mosaic, self.img_size, self.img_size, self.n_channels_all)
        else:
            self.input_shape = (self.img_size, self.img_size, self.n_channels_all)

        self.output_select = tf.constant(tf.range(self.n_output) if output_select is None else output_select)
        self.n_output_select = tf.constant(len(self.output_select), dtype=tf.int64)
        self.scope_name = f"lssnet_{self.netname}"
        
    def save(self, filepath):

        with tf.name_scope(self.scope_name) as scope:

            # store previous checkpoint if exsts
            if self.checkpoint is not None:
                filepath_prev = get_filepath_prev(filepath)
                self.checkpoint.write(filepath_prev)
                utils_io.write_to_pickle(get_filepath_meta(filepath_prev), self.meta_info)
                LOGGER.info(f'stored_checkpoint {filepath_prev}')

            self.checkpoint = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
            self.checkpoint.write(filepath)
            utils_io.write_to_pickle(get_filepath_meta(filepath), self.meta_info)
            LOGGER.info(f'stored_checkpoint {filepath}')


    def load(self, file_in):

        with tf.name_scope(self.scope_name) as scope:

            checkpoint = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
            status = checkpoint.restore(file_in).expect_partial()
            LOGGER.info(f'restored checkpoint {file_in}')

            filepath_aux = f'{file_in}_meta.pkl'        
            meta_info = utils_io.read_from_pickle(filepath_aux)
            self.meta_info.update(meta_info)
            LOGGER.info(f'loaded {file_in}')

    def create_loss(self):

        loss_wrap = losses.LossWrap(output_select=self.output_select, n_output=self.n_output_select)

        input_X = tf.keras.Input(shape=self.input_shape)
        input_y = tf.keras.Input(shape=(self.n_output,))
        inputs = [input_X, input_y]
        predictions = self.model(input_X)
        outputs = loss_wrap(predictions, input_y)
        loss = tf.keras.Model(inputs=inputs, outputs=outputs)
        return loss



    def summary(self):

        self.model.summary()

def get_filepath_meta(filepath):

    return f'{filepath}_meta.pkl' 

def get_filepath_prev(filepath):

    return f'{filepath}_prev'
