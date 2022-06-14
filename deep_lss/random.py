import tensorflow as tf
import numpy as np

@tf.function()
def poisson_sample(lam, shape):
    
    lam_approx = 50 # from here approximate as gaussian
    val_max = 80 # 1e-5 probability for lam=50
    gen_rand = tf.random.get_global_generator()
    u = gen_rand.uniform(shape=shape + (val_max,))
    c = tf.math.cumsum(tf.math.log(u), axis=-1) < -tf.expand_dims(lam, axis=-1)
    p = val_max - tf.reduce_sum(tf.cast(c, tf.float32), axis=-1)
    n = gen_rand.normal(shape=(), mean=lam, stddev=tf.math.sqrt(lam), dtype=tf.dtypes.float32)
    r = tf.where(lam<lam_approx, p, n)

    return tf.math.round(r)

@tf.function()
def anscombe_mean(m): 

    return 2.0 * tf.math.sqrt(m + 0.375) - 0.25/tf.math.sqrt(m)

@tf.function()
def inverse_anscombe(z):
    '''
    https://github.com/broxtronix/pymultiscale/blob/master/pymultiscale/anscombe.py
    Compute the inverse transform using an approximation of the exact unbiased inverse.
    Reference: Makitalo, M., & Foi, A. (2011). A closed-form approximation of the exact unbiased inverse of the Anscombe variance-stabilizing transformation. Image Processing.
    '''
    a = 0.30618621784789724
    b = 0.7654655446197431
    p = 0.25 * tf.math.pow(z, 2)  +  a * tf.math.pow(z, -1.0)  -  1.375 * tf.math.pow(z, -2.0)  +  b * tf.math.pow(z, -3.0)  -  0.125

    return p



@tf.function()
def poisson_anscombe(lam, shape):

    gen_rand = tf.random.get_global_generator()
    
    z = gen_rand.normal(shape=shape, mean=anscombe_mean(lam), stddev=1., dtype=tf.dtypes.float32) 
    p = inverse_anscombe(z)
    p = tf.where(z<1., 0., p)
    p = tf.clip_by_value(p, clip_value_min=0, clip_value_max=1e6) 
    return  p

def perm_sequence(elements, length):

    perms = [np.random.permutation(elements) for _ in range(int(np.ceil(length/len(elements))))]    
    return np.concatenate(perms)[:length]
