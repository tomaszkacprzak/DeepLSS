import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import math

class ResidualLayer(tf.keras.Model):
    """
    A simple residual layer using two convolutions and normalization
    """
    def __init__(self, filters, kernel_size, normalization=None, activation=tf.nn.relu):
        """
        Creates a residual layer that can be used inside networks
        :param filters: Number of filter for the conv, should match the input number of filters
        :param kernel_size: size of the conv kernel to use
        :param normalization: can be None (no normalization), "batch_norm" or "layer_norm"
        :param activation: A TF callable for the activation function
        """

        # this is necessary for all Model subclasses
        super(ResidualLayer, self).__init__(name='')

        # convs
        self.conv1 = tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), padding="same")

        # norm
        if normalization is None:
            self.norm1 = lambda x, training: tf.keras.activations.linear(x)
            self.norm2 = lambda x, training: tf.keras.activations.linear(x)
        elif normalization == "batch_norm":
            self.norm1 = tf.keras.layers.BatchNormalization()
            self.norm2 = tf.keras.layers.BatchNormalization()
        elif normalization == "layer_norm":
            self.norm1 = tf.keras.layers.LayerNormalization()
            self.norm2 = tf.keras.layers.LayerNormalization()
        else:
            raise ValueError(f"Input normalization not understood: {normalization}")

        # activation
        self.activation = activation

    def call(self, input_tensor, training=False):
        """
        Calls the res layer using the input tensor as input
        :param input_tensor: Input of the layer
        :param training: whether we are training or not (relevant for norm)
        :return: output of the layer
        """
        x = self.conv1(input_tensor)
        x = self.norm1(x, training=training)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.activation(x + input_tensor)

        return x




class SelectChannels(tf.keras.layers.Layer):

    def __init__(self, select_start, select_end):

        super(SelectChannels, self).__init__()
        self.select_start = tf.constant(select_start, dtype=tf.int64)
        self.select_end = tf.constant(select_end, dtype=tf.int64)

    def call(self, inputs):

        return inputs[..., self.select_start:self.select_end]


class Smoothing(tf.keras.layers.Layer):

    def __init__(self, sigma, n_channels):

        super(Smoothing, self).__init__()
        self.sigma = sigma
        self.n_channels = n_channels

        if len(self.sigma) == 1:

            """Makes 2D gaussian Kernel for convolution."""
            # https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
            size = tf.cast(5*self.sigma, dtype=tf.int64)
            self.filter_size = (size, size)
            d = tfp.distributions.Normal(0, self.sigma)
            vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
            gauss_kernel = tf.einsum('i,j->ij', vals, vals)
            gauss_kernel /= tf.reduce_sum(gauss_kernel)
            gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
            gauss_kernel = tf.repeat(gauss_kernel, repeats=self.n_channels, axis=3)

        
        else:

            size = tf.cast(5*tf.reduce_max(self.sigma), dtype=tf.int64)
            self.filter_size = (size*2, size*2)

            arr_gauss_kernel = tf.TensorArray(dtype=tf.float32, size=self.n_channels, dynamic_size=False)

            for i in range(self.n_channels):
                d = tfp.distributions.Normal(0, self.sigma[i])
                vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
                gauss_kernel = tf.einsum('i,j->ij', vals, vals)
                gauss_kernel /= tf.reduce_sum(gauss_kernel)
                gauss_kernel = gauss_kernel[:, :, tf.newaxis]
                arr_gauss_kernel = arr_gauss_kernel.write(tf.cast(i, tf.int64), gauss_kernel)

            gauss_kernel = arr_gauss_kernel.stack()
            gauss_kernel = tf.transpose(gauss_kernel, perm=[1,2,3,0])

        self.kernel = gauss_kernel

    def call(self, inputs):

        return tf.nn.conv2d(inputs, self.kernel, strides=1, padding="SAME")    
        # inputs_reshaped = tf.reshape(inputs, shape=(10*16, 128, 128, 4))
        # inputs_smoothed = tfa.image.gaussian_filter2d(inputs_reshaped, filter_shape=self.filter_size, sigma=self.sigma, name='smoothing')
        # return tf.reshape(inputs_smoothed, shape=[10, 16, 128, 128, 4])




class PowerSpectra(tf.keras.layers.Layer):

    def __init__(self, l_edges, n_pix, n_channels, ang, n_ells, stack=False, norm=None):
        """
        Constructor for the power spectrum layer
        :param ang: Angle of the input maps
        :param n_pix: Number of pixels per side of the input maps
        :param l_edges: Edges to sum the spectra
        """

        # with tf.device('gpu'):

        super(PowerSpectra, self).__init__()
        self.n_pix = n_pix
        self.n_channels = n_channels
        self.n_ells = tf.constant(n_ells, dtype=tf.int64)
        self.l_edges = l_edges
        self.l_centers = (self.l_edges[1:]+self.l_edges[:-1])/2.
        self.ang = ang
        self.masks = tf.constant(get_masks(self.ang, self.n_pix, self.l_edges), dtype=tf.float32) # for rfft2
        self.ps_scale = tf.expand_dims(tf.constant(dtype=tf.float32, value=self.l_centers*(self.l_centers+1))/2./math.pi, axis=0)
        self.n_spectra = tf.constant(self.n_channels + (self.n_channels**2-self.n_channels)//2, dtype=tf.int64)
        import numpy as np
        self.tril_indices = tf.constant(np.vstack(np.tril_indices(self.n_channels)).T)
        self.norm = norm
        self.stack = stack

        if self.stack:
            self.call = self.call_vectorised_stack
        else:
            # self.call = self.call_vectorised
            self.call = self.call_serial

    def __call__(self, X):

        return self.call(X)
        # return self.call_vectorised(X)
        # return self.call_serial(X)


    def call_vectorised(self, X):

        Xfft = tf.signal.fft2d(tf.cast(x=tf.transpose(X, perm=[3,0,1,2]), dtype=tf.complex64))
        Xfft_tril1 = tf.gather(Xfft, self.tril_indices[:,0])
        Xfft_tril2 = tf.gather(tf.math.conj(Xfft), self.tril_indices[:,1])
        fft_abs = tf.math.abs(Xfft_tril1*Xfft_tril2)
        power_spec_vec = tf.einsum('cbij,ijk->bkc', fft_abs, self.masks)
        power_spec_vec *= tf.expand_dims(self.ps_scale, axis=-1)
            
        return power_spec_vec


    def call_vectorised_stack(self, X):

        Xfft = tf.signal.fft2d(tf.cast(x=tf.transpose(X, perm=[4,0,1,2,3]), dtype=tf.complex64))
        Xfft = tf.reduce_sum(Xfft, axis=2)
        Xfft_tril1 = tf.gather(Xfft, self.tril_indices[:,0])
        Xfft_tril2 = tf.gather(tf.math.conj(Xfft), self.tril_indices[:,1])
        fft_abs = tf.math.abs(Xfft_tril1*Xfft_tril2)
        power_spec_vec = tf.einsum('cbij,ijk->bkc', fft_abs, self.masks)
        power_spec_vec *= tf.expand_dims(self.ps_scale, axis=-1)
        return power_spec_vec

    def call_serial(self, X):

        power_spec_arr = []
        Xfft = tf.signal.fft2d(tf.cast(x=tf.transpose(X, perm=[3,0,1,2]), dtype=tf.complex64))
        # loop over auto and cross cls combinations
        for i in range(self.n_channels):
            for j in range(i, self.n_channels):

                fft_abs = tf.math.abs(Xfft[i]*tf.math.conj(Xfft[j]))
                power_spec = tf.einsum('bij,ijk->bk', fft_abs, self.masks)
                power_spec *= self.ps_scale
                power_spec_arr.append(tf.expand_dims(power_spec, axis=-1))

        power_spec_arr = tf.concat(power_spec_arr, axis=-1)
        return power_spec_arr





def get_masks(ang, n_pix, l_edges):
    """
    Calculates the masks necessary for the spectra calculation with <get_spectrum>
    :param ang: Angle of the input maps
    :param n_pix: Number of pixels per side of the input maps
    :param l_edges: Edges to sum the spectra
    :return: a numpy array of masks that can be used for <get_spectrum>
    """
    # Get physical pixsize
    lpix = 360.0/ang

    import numpy as np

    # Get the norm
    norm = ((ang*np.pi/180.0)/n_pix**2)**2

    # Get the masks
    masks = []
    tot = 0
    for bins in range(len(l_edges) - 1):
        mask = np.zeros((n_pix, n_pix))

        # Cycle over pixels
        for i in range(n_pix):

            lx = min(i, n_pix-i) * lpix
            #lx = (i - n_pix/2) * lpix

            for j in range(n_pix):

                ly = j*lpix
                l = np.sqrt(lx**2 + ly**2)

                if l_edges[bins] < l <= l_edges[bins + 1]:
                    mask[i,j] += 1.0

        # Normalize
        tot += np.sum(mask)
        mask *= norm/np.sum(mask)
        masks.append(mask)

    return np.stack(masks, axis=-1)








class ResidualLayer1D(tf.keras.Model):
    """
    A simple residual layer using two convolutions and normalization
    """
    def __init__(self, filters, kernel_size, normalization=None, activation=tf.nn.relu):
        """
        Creates a residual layer that can be used inside networks
        :param filters: Number of filter for the conv, should match the input number of filters
        :param kernel_size: size of the conv kernel to use
        :param normalization: can be None (no normalization), "batch_norm" or "layer_norm"
        :param activation: A TF callable for the activation function
        """

        # this is necessary for all Model subclasses
        super(ResidualLayer1D, self).__init__(name='')

        # convs
        self.conv1 = tf.keras.layers.Conv1D(filters, kernel_size, padding="same")
        self.conv2 = tf.keras.layers.Conv1D(filters, kernel_size, padding="same")

        # norm
        if normalization is None:
            self.norm1 = lambda x, training: tf.keras.activations.linear(x)
            self.norm2 = lambda x, training: tf.keras.activations.linear(x)
        elif normalization == "batch_norm":
            self.norm1 = tf.keras.layers.BatchNormalization()
            self.norm2 = tf.keras.layers.BatchNormalization()
        elif normalization == "layer_norm":
            self.norm1 = tf.keras.layers.LayerNormalization()
            self.norm2 = tf.keras.layers.LayerNormalization()
        else:
            raise ValueError(f"Input normalization not understood: {normalization}")

        # activation
        self.activation = activation

    def call(self, input_tensor, training=False):
        """
        Calls the res layer using the input tensor as input
        :param input_tensor: Input of the layer
        :param training: whether we are training or not (relevant for norm)
        :return: output of the layer
        """
        x = self.conv1(input_tensor)
        x = self.norm1(x, training=training)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.activation(x + input_tensor)

        return x



class FieldsMosaic(tf.keras.layers.Layer):

    def __init__(self, n_fields, n_pix, n_channels):

        super(FieldsMosaic, self).__init__()
        self.n_fields = n_fields
        self.mosaic_size = tf.constant(tf.cast(tf.sqrt(tf.cast(n_fields, dtype=tf.float32)), dtype=tf.int64))
        self.tile_size = n_pix
        self.n_channels = n_channels
        self.pic_size = self.tile_size * self.mosaic_size
        

    def call(self, X):

        if self.n_fields==1: # no mosaic
            Xm = X[0]

        else:
            # kick out batch dimension to the end
            Xm = tf.transpose(X, perm=[1, 2, 3, 4,0])
            # https://stackoverflow.com/questions/65623486/tensorflow-assemble-tiles-into-mosaic-image
            Xm = tf.reshape(Xm, shape=[self.mosaic_size, self.mosaic_size, self.tile_size, self.tile_size, self.n_channels, -1])
            Xm = tf.transpose(Xm, perm=[0, 2, 1, 3, 4, 5])
            Xm = tf.reshape(Xm, shape=(self.pic_size, self.pic_size, self.n_channels, -1))
            # bring batch dim to the start
            Xm = tf.transpose(Xm, perm=[3, 0, 1, 2])
        
        return Xm


class FlipFields(tf.keras.layers.Layer):

    def __init__(self, n_fields):

        super(FlipFields, self).__init__()
        self.n_fields = n_fields

    def call(self, X):

        # with tf.device('gpu'):

        Xf = tf.TensorArray(dtype=tf.float32, size=self.n_fields, dynamic_size=False, infer_shape=True)

        # random flips
        for i in range(self.n_fields):
            X_ = tf.image.random_flip_left_right(tf.image.random_flip_up_down(X[:,i,...]))
            Xf = Xf.write(i, X_)

        Xf = tf.transpose(Xf.stack(), perm=[1,0,2,3,4])
        return Xf

