import tensorflow as tf
from deep_lss.networks import layers
from deep_lss.networks import losses
from deep_lss import utils_logging, utils_io
LOGGER = utils_logging.get_logger(__file__)
from deep_lss.networks.lssnet import *

class MultiLssNET():

    def __init__(self, nets, tag='model'):

        self.nets = nets
        self.tag = tag
        
        self.models = [net.model for net in self.nets]
        self.n_nets = len(self.nets)
        self.input_shape = self.nets[0].input_shape
        self.n_output = self.nets[0].n_output 

        self.scope_name = f'multilssnet_n{self.n_nets}_{tag}'
        self.opts = [net.opt for net in self.nets]

        self.checkpoint = None
        # https://kipoi.org/docs/tutorials/composing_models/
        self.meta_info = {'step':0, 'epoch':0}
        self.create_network()


    def create_network(self):


        layer_input_X = tf.keras.Input(shape=self.input_shape, 
                                       name='multimodel_input_X')

        outputs = [net.model(layer_input_X) for net in self.nets]

        self.model = tf.keras.Model(inputs=layer_input_X,
                                    outputs=outputs,
                                    name=f'model_{self.scope_name}')

        layer_input_y = tf.keras.Input(shape=(self.n_output,), name='multimodel_input_y')
        inputs = [layer_input_X, layer_input_y]

        loss_total = []
        loss_parts = []
        errs = []

    
        loss = [net.loss(inputs=inputs) for net in self.nets]

        loss_total = [l[0] for l in loss]
        loss_parts = [[l[0], l[1], l[2]] for l in loss]
        err = [l[3] for l in loss]
        # loss_multi = tf.keras.layers.Add(name='multimodel_add_losses')(loss_total)

        self.loss = tf.keras.Model(inputs=inputs,
                                   outputs=(loss_total, loss_parts, err),
                                   name=f'loss_{self.scope_name}')

    def summary(self):

        self.model.summary()

