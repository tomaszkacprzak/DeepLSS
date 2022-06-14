import tensorflow as tf
from deep_lss.networks import layers
from deep_lss.networks import losses
from deep_lss.networks.lssnet import LssNET
from deep_lss import utils_logging, utils_io
LOGGER = utils_logging.get_logger(__file__)

class_name = 'LssCNN'

class LssCNN(LssNET):

    def __init__(self, netname='lsscnn', **kwargs):
        """
        :param smoothing_sigma: Apply Gaussian smoothing with sigma (pixels)
        """
        super().__init__(netname=netname, **kwargs)

        self.nn_config = {}
        self.nn_config.setdefault('filters', 128)
        self.nn_config.setdefault('kernel_size', 3)
        self.nn_config.setdefault('strides', (2,2))
        self.nn_config.setdefault('n_conv_layers', 3)
        self.nn_config.setdefault('n_res_layers', 10)
        self.nn_config.setdefault('activation', 'relu')
        self.nn_config.setdefault('normalization', None)
        self.nn_config.setdefault('downsample', None)
        self.nn_config.setdefault('orig_dim', True)

        # update nn_config keys
        for k in (set(kwargs.keys()) & set(self.nn_config.keys())):
            self.nn_config[k] = kwargs[k]

        self.model = self.create_network()
        self.loss = self.create_loss()

    def create_network(self):   

            with tf.name_scope(f"lssnet_{self.netname}") as scope:

                layers_input = [tf.keras.Input(shape=self.input_shape)]

                if not self.all_channels:
                    layers_input += [layers.SelectChannels(select_start=self.channels_select_start,
                                                           select_end=self.channels_select_end)]

                layer_smoothing = [layers.Smoothing(sigma=self.smoothing_sigma, 
                                                    n_channels=self.n_channels)]


                if self.mosaic > 1:
                    layer_smoothing += [layers.FieldsMosaic(n_fields=self.mosaic, 
                                                            n_pix=self.img_size, 
                                                            n_channels=self.n_channels)]

                if self.nn_config['downsample'] is not None:
                    layer_smoothing += [tf.keras.layers.AveragePooling2D(pool_size=(self.nn_config['downsample'], self.nn_config['downsample']))]
                    if self.nn_config['orig_dim']:
                        layer_smoothing += [tf.keras.layers.UpSampling2D(size=(self.nn_config['downsample'], self.nn_config['downsample']), 
                                                                         interpolation='nearest')]
                layers_conv = []
                for i in range(self.nn_config['n_conv_layers']):

                    layers_conv += [tf.keras.layers.Conv2D(self.nn_config['filters'], 
                                                          self.nn_config['kernel_size'], 
                                                          strides=self.nn_config['strides'], 
                                                          padding="same", 
                                                          activation=self.nn_config['activation'])]
                    
                    if self.nn_config['normalization'] == 'batch_norm':
                        layers_conv += [tf.keras.layers.BatchNormalization()]

                    elif self.nn_config['normalization'] == 'layer_norm':
                        layers_conv += [tf.keras.layers.LayerNormalization()]


                layers_res = [layers.ResidualLayer(self.nn_config['filters'], 
                                                   self.nn_config['kernel_size'], 
                                                   normalization=self.nn_config['normalization'], 
                                                   activation=tf.keras.activations.get(self.nn_config['activation'])) 
                                                            for _ in range(self.nn_config['n_res_layers'])]

                layers_dense = [tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(self.n_output_select + int(self.n_output_select * (self.n_output_select + 1) / 2))]
                
                self.network_imgs = tf.keras.Sequential(layers_input + layer_smoothing, name=f'imgs_{self.netname}')
                network = tf.keras.Sequential(layers_input + layer_smoothing + layers_conv + layers_res + layers_dense, name=self.netname)

            return network




