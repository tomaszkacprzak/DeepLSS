import tensorflow as tf, numpy as np
from deep_lss.networks import layers
from deep_lss.networks import losses
from deep_lss.networks.lssnet import LssNET
from deep_lss import utils_logging, utils_io
LOGGER = utils_logging.get_logger(__file__)

class_name = 'LssPSD'

class LssPSD(LssNET):

    def __init__(self, angle, ells, stack=False, netname='lsspsd', **kwargs):
        """
        :param smoothing_sigma: Apply Gaussian smoothing with sigma (pixels)
        """
        super().__init__(netname=netname, **kwargs)

        def get_ell_bins(ell_min, ell_max, n_ells):
            ell_all = np.fft.fftfreq(n=self.ps_config['n_pix'], d=np.radians(self.ps_config['ang'])/self.ps_config['n_pix'])*np.pi
            ell_min_img, ell_max_img = int(ell_all[1]), int(2*np.max(np.abs(ell_all)))
            ell_max = min(ell_max, ell_max_img)
            ell_delta = 36 # this is the smallest delta we can use for 5deg/128pix
            fftfreqs = np.arange(ell_min_img, ell_max_img+1, ell_min_img)
            fftfreqs = fftfreqs[fftfreqs%ell_delta==0] # filter out the ones that can be divided by ell_delta
            LOGGER.debug(f'ells={fftfreqs[0]}:{fftfreqs[-1]}:{fftfreqs[0]}')
            if len(fftfreqs)<n_ells:
                ell_edges = fftfreqs

            else:
                for i in range(n_ells, ell_max_img//ell_delta-1):
                    edges_propose = np.logspace(np.log10(ell_min), np.log10(ell_max), i+1)
                    idx = np.searchsorted(fftfreqs, edges_propose)
                    ell_edges = np.sort(np.unique(fftfreqs[idx]))
                    if len(ell_edges) == (n_ells+1):
                        break
            return ell_edges
        

        self.nn_config = {}
        self.nn_config.setdefault('n_units', 128)
        self.nn_config.setdefault('n_hidden_layers', 3)
        self.nn_config.setdefault('activation', 'relu')
        self.nn_config.setdefault('filters', 128)
        self.nn_config.setdefault('kernel_size', 3)
        self.nn_config.setdefault('n_conv_layers', 3)
        self.nn_config.setdefault('n_res_layers', 10)
        self.nn_config.setdefault('normalization', None)
        self.nn_config.setdefault('strides', 1)
        self.nn_config.setdefault('downsample', None)
        self.nn_config.setdefault('orig_dim', True)
        self.nn_config.setdefault('arch', 'separable')
        self.nn_config.setdefault('psd_normalization', False)

        # update nn_config keys
        for k in (set(kwargs.keys()) & set(self.nn_config.keys())):
            self.nn_config[k] = kwargs[k]

        self.ps_config = {}
        self.ps_config['stack'] = stack           
        self.ps_config['fields_mosaic'] = (self.mosaic>1) and (self.ps_config['stack']==False) # decide if to make a mosaic or not 
    
        if self.ps_config['fields_mosaic']: # make mosaic of fields        
        
            self.ps_config['ang'] = angle*int(self.mosaic_side)
            self.ps_config['n_pix'] = int(self.img_size*self.mosaic_side) if self.nn_config['orig_dim'] else int(self.img_size*self.mosaic_side//self.nn_config['downsample'])

        else: # retain the original image size, no mosaic of fields

            self.ps_config['ang'] = angle
            self.ps_config['n_pix'] = self.img_size if self.nn_config['orig_dim'] else int(self.img_size//self.nn_config['downsample'])

        self.ps_config['l_edges'] = get_ell_bins(*ells)
        self.ps_config['l_centers'] = (self.ps_config['l_edges'][:-1]+self.ps_config['l_edges'][1:])/2.
        self.ps_config['n_ells'] = len(self.ps_config['l_centers'])
        LOGGER.info(f"ang={self.ps_config['ang']} n_pix={self.ps_config['n_pix']} l_edges={np.array2string(self.ps_config['l_edges'], max_line_width=int(1e5))} ({self.ps_config['n_ells']} bins)")
        LOGGER.info(f"mosaic={self.mosaic} stack={self.ps_config['stack']} fields_mosaic={self.ps_config['fields_mosaic']}")

        self.n_spectra = self.n_channels + (self.n_channels**2-self.n_channels)//2
        self.input_shape_ps = self.n_spectra*self.ps_config['n_ells']

        # get network        
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

            if self.ps_config['fields_mosaic']:
                layer_smoothing += [layers.FieldsMosaic(n_fields=self.mosaic, 
                                                        n_pix=self.img_size, 
                                                        n_channels=self.n_channels)]

            if self.nn_config['downsample'] is not None:
                layer_smoothing += [tf.keras.layers.AveragePooling2D(pool_size=(self.nn_config['downsample'], self.nn_config['downsample']))]
                if self.nn_config['orig_dim']:
                    layer_smoothing += [tf.keras.layers.UpSampling2D(size=(self.nn_config['downsample'], self.nn_config['downsample']), 
                                                                     interpolation='nearest')]

            layer_powspec = [layers.PowerSpectra(stack=self.ps_config['stack'],
                                                 l_edges=self.ps_config['l_edges'], 
                                                 n_pix=self.ps_config['n_pix'], 
                                                 ang=self.ps_config['ang'], 
                                                 n_channels=self.n_channels,
                                                 n_ells=self.ps_config['n_ells'])]
            
            if self.nn_config['psd_normalization']:
                layer_powspec += [tf.keras.layers.LayerNormalization()]


            layers_nn = []
            for i in range(self.nn_config['n_conv_layers']):

                if self.nn_config['arch'] == 'separable':

                    layers_nn += [tf.keras.layers.SeparableConv1D(filters=self.nn_config['filters'], 
                                                                   kernel_size=self.nn_config['kernel_size'], 
                                                                   strides=self.nn_config['strides'], 
                                                                   padding="same", 
                                                                   activation=self.nn_config['activation'])]
         
                elif self.nn_config['arch'] == 'local':
                    
                    layers_nn += [tf.keras.layers.LocallyConnected1D(self.nn_config['filters'], 
                                                                      self.nn_config['kernel_size'], 
                                                                      strides=self.nn_config['strides'], 
                                                                      padding="valid", 
                                                                      activation=self.nn_config['activation'])]

                elif self.nn_config['arch'] == 'classic':

                    #non-conv nn
                    if i == 0:
                        layers_nn += [tf.keras.layers.Flatten()]
                    layers_nn += [tf.keras.layers.Dense(self.nn_config['filters'], activation=self.nn_config['activation'])]
                
                elif self.nn_config['arch'] == 'conv':

                    layers_nn += [tf.keras.layers.Conv1D(self.nn_config['filters'], 
                                                          self.nn_config['kernel_size'], 
                                                          strides=self.nn_config['strides'], 
                                                          padding="same", 
                                                          activation=self.nn_config['activation'])]
                else:
                    raise Exception(f"unknown architecture={self.nn_config['arch']}")

                # add normalization
                if self.nn_config['normalization'] == 'batch_norm':
                    layers_conv += [tf.keras.layers.BatchNormalization()]

                elif self.nn_config['normalization'] == 'layer_norm':
                    layers_conv += [tf.keras.layers.LayerNormalization()]


            # resnet
            layers_res = [layers.ResidualLayer1D(self.nn_config['filters'], 
                                                 self.nn_config['kernel_size'], 
                                                 normalization=self.nn_config['normalization'], 
                                                 activation=tf.keras.activations.get(self.nn_config['activation'])) 
                                                        for _ in range(self.nn_config['n_res_layers'])]

            layers_dense = [tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(self.n_output_select + int(self.n_output_select * (self.n_output_select + 1) / 2))]

            self.network_imgs = tf.keras.Sequential(layers_input + layer_smoothing, name=f'imgs_{self.netname}')
            self.network_psds = tf.keras.Sequential(layers_input + layer_smoothing + layer_powspec, name=f'psds_{self.netname}')
            
            network = tf.keras.Sequential(layers_input + layer_smoothing + layer_powspec + layers_nn + layers_res + layers_dense, name=self.netname)

        return network










