import h5py, numpy as np, tensorflow as tf, itertools, warnings
from deep_lss.utils_weights_calc import cosmo_rename
from deep_lss.apps.run_mapmaker import get_filepath_kappa_maps
from deep_lss import utils_logging, utils_io, utils_weights_calc, utils_arrays
from functools import partial
from deep_lss.random import poisson_anscombe as poisson_sample
import tensorflow_probability as tfp
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)


# kappa_maps.shape == (128, 128, 3, 4)
# kappa_maps[:,:,0,:] = convergence
# kappa_maps[:,:,1,:] = intrinsic alignment
# kappa_maps[:,:,2,:] = galaxy density

IND_KG = 0 
IND_IA = 1
IND_DG = 2


def get_param_fid_lims(prior_lims, theta):

    fid = np.array([prior_lims[p][0] if type(prior_lims[p]) is list else prior_lims[p] for p in theta])
    lim_lo = np.array([prior_lims[p][1] if type(prior_lims[p]) is list else 0 for p in theta])
    lim_hi = np.array([prior_lims[p][2] if type(prior_lims[p]) is list else 1 for p in theta])
    fixed = np.array([False if type(prior_lims[p]) is list else True for p in theta], dtype=np.bool)
    var = tf.expand_dims(tf.constant(~fixed, dtype=tf.float32), axis=0)
    fid = tf.expand_dims(tf.constant(fixed*fid, dtype=tf.float32), axis=0)
    lim_range = tf.expand_dims(tf.constant(lim_hi-lim_lo, dtype=tf.float32), axis=0)
    lim_min = tf.expand_dims(tf.constant(lim_lo, dtype=tf.float32), axis=0)
    return var, fid, lim_min, lim_range



class LssAstrophysicsModel():

    def __init__(self, params_astro_const, theta_priors, probes, nz, img_size):

        self.theta_priors = theta_priors
        self.probes = probes
        self.nz = tf.constant(nz, dtype=tf.float32)

        # constant variables
        self.map_size_deg = params_astro_const['map_patch_size_deg']
        self.ng_eff = params_astro_const['ng_eff']
        self.sigma_e = params_astro_const['sigma_e']
        self.z_ia_pivot = params_astro_const['z_ia_pivot']
        self.z_bg_pivot = params_astro_const['z_bg_pivot']
        self.available_galaxy_counts_noise_models = ['baseline', 'stochasticity', 'super-poissonian', 'no_clustering_for_kg', 'nonlin_bias']
        self.galaxy_counts_noise_model = params_astro_const['galaxy_counts_noise_model']
        assert self.galaxy_counts_noise_model in self.available_galaxy_counts_noise_models, f'unknown galaxy counts noise model {self.galaxy_counts_noise_model}'
        LOGGER.info(f'using galaxy counts noise model: {self.galaxy_counts_noise_model}')

        # galaxy density and shape noise calculation
        self.patch_area_arcmin2 = (self.map_size_deg*60.)**2
        self.pixel_size_arcmin = (self.map_size_deg*60.)/img_size
        self.pixel_area_arcmin2 = self.pixel_size_arcmin**2
        self.n_gal_per_pix = self.ng_eff * self.pixel_area_arcmin2
        self.pixel_sig = self.sigma_e/np.sqrt(self.n_gal_per_pix)
        self.scale_kg = [1./np.sqrt(0.005**2 + self.pixel_sig**2), 0]
        self.scale_dg = [1./self.n_gal_per_pix, -1]
        LOGGER.info(f'n_gal_per_pix={self.n_gal_per_pix:2.4f} pixel_sig={self.pixel_sig:2.4f}')


    def get_func_scale_astro_to_prior_range(self):

        var, fid, lim_min, lim_range = get_param_fid_lims(prior_lims=self.theta_priors.prior_lims, 
                                                          theta=self.theta_priors.theta_full_dynamic)

        @tf.function()
        def scale_to_prior_range(y):

            y *= lim_range
            y += lim_min
            y *= var 
            y += fid
            
            return y

        return scale_to_prior_range

    def get_func_theta_transform_forward(self, method='none'):

        if method == 'unit':

            var, fid, lim_min, lim_range = get_param_fid_lims(prior_lims=self.theta_priors.prior_lims, 
                                                              theta=self.theta_priors.theta_full)
            
            @tf.function()
            def transform_forward(y):
                y -= lim_min
                y /= lim_range
                y -= 0.5
                y *= var 
                y += fid
                return y


        elif method == 'S8':

            @tf.function()
            def transform_forward(y):

                y_list = tf.unstack(y, axis=1)

                # self.theta_full = ['Omega_m', 'sigma_8', 'A_IA', 'b_gal', 'r_stoch', 'alpha0_spoiss', 'alpha1_spoiss', 'eta_A_IA', 'eta_b_gal', 'b2_gal', 'eta_b2_gal'] # all parameters that can be varied in this model
                y_list[1] *= tf.sqrt(y_list[0]/0.3)
                y_list[2] *= 0.1 

                return tf.stack(y_list, axis=1)

        elif method == 'none':

            @tf.function()
            def transform_forward(y):
                return y

        else:

            raise Exception(f'unknown theta transform method {method}')

        return transform_forward




    def get_func_theta_transform_inverse(self, method='none'):


        if method == 'unit':
            
            var, fid, lim_min, lim_range = get_param_fid_lims(prior_lims=self.theta_priors.prior_lims, 
                                                              theta=self.theta_priors.theta_full)
    
            @tf.function()
            def transform_inverse(y, indices):
                lim_range_select = tf.gather(lim_range, indices, axis=1)
                lim_min_select = tf.gather(lim_min, indices, axis=1)
                var_select = tf.gather(var, indices, axis=1)
                fid_select = tf.gather(fid, indices, axis=1)
                y += 0.5
                y *= lim_range_select
                y += lim_min_select
                y *= var_select 
                y += fid_select
                return y

        elif method == 'S8':
            
            n_theta_full = len(self.theta_priors.theta_full)

            @tf.function()
            def transform_inverse(y, indices):

                ys = tf.TensorArray(dtype=tf.float32, size=y.shape[0], dynamic_size=False)
                for i in range(y.shape[0]):
                    ys = ys.write(i, tf.scatter_nd(tf.expand_dims(indices, axis=1), y[i], shape=(n_theta_full,)))
                ys = ys.stack()

                y_list = tf.unstack(ys, axis=1)
                y_list[1] /= tf.sqrt(y_list[0]/0.3)
                y_list[2] *= 10 
                ys = tf.stack(y_list, axis=1)
                y = tf.gather(ys, indices, axis=1)
                return y

        elif method == 'none':

            @tf.function()
            def transform_inverse(y, indices):
                return y

        else:

            raise Exception(f'unknown theta transform method {method}')

        return transform_inverse



    def get_sampler_astro(self, batch_size):

        gen_rand = tf.random.get_global_generator()
        lim_lo = [float(self.theta_priors.prior_lims[p][0]) for p in self.theta_priors.theta_full_dynamic]
        lim_hi = [float(self.theta_priors.prior_lims[p][1]) for p in self.theta_priors.theta_full_dynamic]
        lim_lo = tf.constant(lim_lo, dtype=np.float32)
        lim_hi = tf.constant(lim_hi, dtype=np.float32)
        lim_lo = tf.expand_dims(lim_lo, axis=0)
        lim_hi = tf.expand_dims(lim_hi, axis=0)
        n_theta_add = tf.constant(len(self.theta_priors.theta_full_dynamic), dtype=tf.int64)
        batch_size = tf.constant(batch_size, dtype=tf.int64)

        @tf.function()
        def generator_astro():
            
            u = gen_rand.uniform(shape=(batch_size, n_theta_add), minval=lim_lo, maxval=lim_hi, dtype=tf.dtypes.float32)
            return u

        return generator_astro


        

    def get_fun_process_dataset(self, n_fields, n_pix, n_maps, n_zbins, batch_size, transform_maps=False):

        # with tf.device('gpu'):

        dims = tf.constant([n_pix, n_pix, n_zbins, batch_size, n_fields], dtype=tf.int64)
        noise = tf.constant([self.sigma_e, self.n_gal_per_pix], dtype=tf.float32)
        zevol = tf.constant([self.z_ia_pivot, self.z_bg_pivot], dtype=tf.float32)
        nzg = self.nz

        if self.galaxy_counts_noise_model == 'baseline':
            preprocess_dataset = process_dataset_bas

        elif self.galaxy_counts_noise_model == 'stochasticity':
            preprocess_dataset = process_dataset_sto

        elif self.galaxy_counts_noise_model == 'super-poissonian':
            preprocess_dataset = process_dataset_sup

        elif self.galaxy_counts_noise_model == 'no_clustering_for_kg':
            preprocess_dataset = process_dataset_noc

        elif self.galaxy_counts_noise_model == 'nonlin_bias':
            preprocess_dataset = process_dataset_sb2

        else:
            raise Exception('failed to get processing function for dataset')

        @tf.function()
        def map_func(X, y, n_gal_per_pix):

                # with tf.device('gpu'):
                    
                    return preprocess_dataset(X, y, n_gal_per_pix, noise, zevol, nzg, dims, transform_maps)

        return map_func

    def process_targets(self, theta):
        """
        Parameter ordering for all models/probes cases
        """
        return np.atleast_2d([theta[t] for t in self.theta_priors.theta_full]).transpose()


# map ordering
# 0: kappa lensing
# 1: intrinsic alignment
# 2: galaxy counts

# full params ordering:
# 0 omega_m
# 1 sigma_8
# 2 A_IA
# 3 b_gal
# 4 r_stoch
# 5 alpha0_spoiss
# 6 alpha1_spoiss
# 7 eta_A_IA
# 8 eta_b_gal 

@tf.function()
def get_mean_zevol(z0, eta, nz, zg, n_zbins, batch_size):
    """
    Get redshift evolution of a quantity
    """

    eta_e = tf.reshape(eta, shape=(batch_size, 1, 1))
    zg_e = tf.expand_dims(zg, axis=0)
    f = ((1+zg)/(1+z0))**eta_e
    fz = tf.reduce_sum(nz*f, axis=2)/tf.expand_dims(tf.reduce_sum(nz, axis=1), axis=0)

    return fz

@tf.function()
def process_dataset_bas(X, y, n_gal_per_pix, noise, zevol, nzg, dims, transform_maps):    
    """
    Get dataset with baseline bias
    """

    map_dg = get_map_dg_bas(X, y, n_gal_per_pix, zevol, nzg, dims)  
    map_kg = get_map_kg(X, y, map_dg, noise, zevol, nzg, dims)    

    sigma_e = noise[0]
    if transform_maps:
        Xm =  tf.concat([scale_kg(map_kg, n_gal_per_pix, sigma_e), 
                         scale_dg(map_dg, n_gal_per_pix)], 
                         axis=-1)
    else:
        Xm =  tf.concat([map_kg, map_dg], axis=-1)

    return Xm


@tf.function()
def process_dataset_noc(X, y, n_gal_per_pix, noise, zevol, nzg, dims, transform_maps):    
    """
    Get dataset with baseline bias, and no clustering for lensing (for comparison to old results)
    """

    map_dg = get_map_dg_bas(X, y, n_gal_per_pix, zevol, nzg, dims)  
    map_kg = get_map_kg(X, y, n_gal_per_pix, noise, zevol, nzg, dims)    

    sigma_e = noise[0]
    if transform_maps:
        Xm =  tf.concat([scale_kg(map_kg, n_gal_per_pix, sigma_e), 
                         scale_dg(map_dg, n_gal_per_pix)], 
                         axis=-1)
    else:
        Xm =  tf.concat([map_kg, map_dg], axis=-1)

    return Xm

@tf.function()
def process_dataset_sto(X, y, n_gal_per_pix, noise, zevol, nzg, dims, transform_maps):    
    """
    Get dataset with stochasticity
    """

    map_dg = get_map_dg_sto(X, y, n_gal_per_pix, zevol, nzg, dims)  
    map_kg = get_map_kg(X, y, map_dg, noise, zevol, nzg, dims)    

    sigma_e = noise[0]
    if transform_maps:
        Xm =  tf.concat([scale_kg(map_kg, n_gal_per_pix, sigma_e), 
                         scale_dg(map_dg, n_gal_per_pix)], 
                         axis=-1)
    else:
        Xm =  tf.concat([map_kg, map_dg], axis=-1)

    return Xm


@tf.function()
def process_dataset_sb2(X, y, n_gal_per_pix, noise, zevol, nzg, dims, transform_maps):    
    """
    Get dataset with stochasticity and non-linear bias 
    """

    map_dg = get_map_dg_sb2(X, y, n_gal_per_pix, zevol, nzg, dims)  
    map_kg = get_map_kg(X, y, map_dg, noise, zevol, nzg, dims)    

    sigma_e = noise[0]
    if transform_maps:
        Xm =  tf.concat([scale_kg(map_kg, n_gal_per_pix, sigma_e), 
                         scale_dg(map_dg, n_gal_per_pix)], 
                         axis=-1)
    else:
        Xm =  tf.concat([map_kg, map_dg], axis=-1)

    return Xm


@tf.function()
def process_dataset_sup(X, y, n_gal_per_pix, noise, zevol, nzg, dims, transform_maps):    
    """
    Get dataset with super-poisson noise model
    """
    
    map_dg = get_map_dg_sup(X, y, n_gal_per_pix, zevol, nzg, dims)  
    map_kg = get_map_kg(X, y, map_dg,  noise, zevol, nzg, dims)    

    sigma_e = noise[0]
    if transform_maps:
        Xm =  tf.concat([scale_kg(map_kg, n_gal_per_pix, sigma_e), 
                         scale_dg(map_dg, n_gal_per_pix)], 
                         axis=-1)
    else:
        Xm =  tf.concat([map_kg, map_dg], axis=-1)


    return Xm


@tf.function()
def get_map_kg(X, y, map_dg, noise, zevol, nzg, dims):    
    """
    Lensing map with noise and IA
    """

    gen_rand = tf.random.get_global_generator()
    n_pix = dims[0]
    n_zbins = dims[2]
    batch_size = dims[3]
    n_fields = dims[4]
    zg = nzg[:,:,0]
    nz = nzg[:,:,1]
    z_pivot = zevol[0]
    eta = y[:,7]
    sigma_e = noise[0]
    
    kappa_g = image_mean_subtract(X[:,:,:,:,0,:])
    kappa_i = image_mean_subtract(X[:,:,:,:,1,:])

    A_IA = tf.expand_dims(y[:,2], axis=-1)
    A_IA_evol = A_IA * get_mean_zevol(z_pivot, eta, nz, zg, n_zbins, batch_size)
    A_IA_evol = tf.reshape(A_IA_evol, shape=(batch_size, 1, 1, 1, n_zbins))
    map_kg = kappa_g + A_IA_evol * kappa_i

    map_sig = sigma_e/tf.math.sqrt(map_dg+1.0)
    map_kg += gen_rand.normal(shape=(batch_size, n_fields, n_pix, n_pix, n_zbins))*map_sig

    # tf.debugging.assert_all_finite(map_kg, message='found not finite values in map_kg')

    return map_kg

@tf.function()
def get_map_dg_bas(X, y, n_gal_per_pix, zevol, nzg, dims):    
    """
    Baseline bias model
    """

    n_pix = dims[0]
    n_zbins = dims[2]
    batch_size = dims[3]
    n_fields = dims[4]
    zg = nzg[:,:,0]
    nz = nzg[:,:,1]
    z_pivot = zevol[1]
    b_gal = tf.expand_dims(y[:,3], axis=-1) 
    eta = y[:,8]

    map_dm = density_contrast(X[:,:,:,:,2,:])
    
    b_gal_evol = b_gal * get_mean_zevol(z_pivot, eta, nz, zg, n_zbins, batch_size)
    b_gal_evol = tf.reshape(b_gal_evol, shape=(batch_size, 1, 1, 1, n_zbins))
    poisson_lam_base = n_gal_per_pix * (1 + b_gal_evol*map_dm)
    poisson_lam_sum = tf.reduce_sum(poisson_lam_base, axis=(2,3), keepdims=True)
    poisson_lam_clip = tf.clip_by_value(poisson_lam_base, clip_value_min=0, clip_value_max=1e5)
    poisson_lam_clip_sum = tf.reduce_sum(poisson_lam_clip, axis=(2,3), keepdims=True)
    poisson_lam = poisson_lam_clip * poisson_lam_sum / poisson_lam_clip_sum # normalize to keep the number counts
    map_dg = poisson_sample(lam=poisson_lam, shape=(batch_size, n_fields, n_pix, n_pix, n_zbins))

    # tf.debugging.assert_all_finite(map_dg, message='found not finite values in map_dg')

    return map_dg

@tf.function()
def get_map_dg_sto(X, y, n_gal_per_pix, zevol, nzg, dims):
    """
    Stochasticity with random phases
    """

    n_pix = dims[0]
    n_zbins = dims[2]
    batch_size = dims[3]
    n_fields = dims[4]
    zg = nzg[:,:,0]
    nz = nzg[:,:,1]
    z_pivot = zevol[1]
    eta = y[:,8]
    b_gal = tf.expand_dims(y[:,3], axis=-1) 
    
    # remove mean from maps
    map_dm = density_contrast(X[:,:,:,:,2,:])
        
    # get galaxy density maps
    b_gal_evol = b_gal * get_mean_zevol(z_pivot, eta, nz, zg, n_zbins, batch_size)
    b_gal_evol = tf.reshape(b_gal_evol, shape=(batch_size, 1, 1, 1, n_zbins))
    map_bdm = map_dm * b_gal_evol
    
    r_stoch = tf.reshape(y[:,4], shape=(batch_size, 1, 1, 1, 1))
    m_prime = add_phase_noise(map_bdm, r_stoch, n_pix, batch_size, n_fields, n_zbins)
    poisson_lam_base = n_gal_per_pix * (1 + m_prime)
    map_dg = poisson_sample_clipped(poisson_lam_base, shape=(batch_size, n_fields, n_pix, n_pix, n_zbins))

    # tf.debugging.assert_all_finite(map_dg, message='found not finite values in map_dg')

    return map_dg

@tf.function()
def get_map_dg_sb2(X, y, n_gal_per_pix, zevol, nzg, dims):
    """
    Stochasticity model with second order bias
    """

    n_pix = dims[0]
    n_zbins = dims[2]
    batch_size = dims[3]
    n_fields = dims[4]
    zg = nzg[:,:,0]
    nz = nzg[:,:,1]
    z_pivot = zevol[1]
    b_gal = tf.expand_dims(y[:,3], axis=-1) 
    b2_gal = tf.expand_dims(y[:,9], axis=-1)
    eta_b_gal = y[:,8]
    eta_b2_gal = y[:,10]

    map_dm = density_contrast(X[:,:,:,:,2,:])
    
    b_gal_evol = b_gal * get_mean_zevol(z_pivot, eta_b_gal, nz, zg, n_zbins, batch_size)
    b_gal_evol = tf.reshape(b_gal_evol, shape=(batch_size, 1, 1, 1, n_zbins))

    b2_gal_evol = b2_gal * get_mean_zevol(z_pivot, eta_b2_gal, nz, zg, n_zbins, batch_size)
    b2_gal_evol = tf.reshape(b2_gal_evol, shape=(batch_size, 1, 1, 1, n_zbins))

    map_dg = map_dm * b_gal_evol +  0.5 * tf.math.sign(map_dm)*tf.square(map_dm) * b2_gal_evol
    map_dg = image_mean_subtract(map_dg)
    
    r_stoch = tf.reshape(y[:,4], shape=(batch_size, 1, 1, 1, 1))
    m_prime = add_phase_noise(map_dg, r_stoch, n_pix, batch_size, n_fields, n_zbins)
    poisson_lam_base = n_gal_per_pix * (1 + m_prime)
    map_dg = poisson_sample_clipped(poisson_lam_base, shape=(batch_size, n_fields, n_pix, n_pix, n_zbins))

    # tf.debugging.assert_all_finite(map_dg, message='found not finite values in map_dg')

    return map_dg

@tf.function()
def get_map_dg_sup(X, y, n_gal_per_pix, zevol, nzg, dims):  
    """
    Super-poissonian model
    """
    

    gen_rand = tf.random.get_global_generator()
    # unpack
    n_pix = dims[0]
    n_zbins = dims[2]
    batch_size = dims[3]
    n_fields = dims[4]
    zg = nzg[:,:,0]
    nz = nzg[:,:,1]
    z_pivot = zevol[1]
    b_gal = y[:,3] 
    eta = y[:,8]

    map_dm = density_contrast(X[:,:,:,:,2,:])

    #arXiv:1710.05045v2 eqn III.16
    alpha0 = tf.reshape(y[:,5], shape=(batch_size, 1, 1, 1, 1))
    alpha1 = tf.reshape(y[:,6], shape=(batch_size, 1, 1, 1, 1))
    alpha = alpha0 + map_dm*alpha1

    # get evolving bias
    b_gal_evol = b_gal * get_mean_zevol(z_pivot, eta, nz, zg, n_zbins, batch_size)
    b_gal_evol = tf.reshape(b_gal_evol, shape=(batch_size, 1, 1, 1, n_zbins))

    # clip to prevent division by very small numbers
    alpha = tf.clip_by_value(alpha, clip_value_min=0.1, clip_value_max=1e200)

    # draw renormalisation to keep the total counts on the right level (0-th mode in power spectra)
    n_gal_total = n_gal_per_pix * dims[0] * dims[1]
    poisson_total = gen_rand.normal(mean=n_gal_total, stddev=tf.math.sqrt(n_gal_total)) # gaussian approx to poisson

    # get poisson lambda and clip
    poisson_lam = n_gal_per_pix * (1 + b_gal*map_dm) / alpha
    tf.clip_by_value(poisson_lam, clip_value_min=0, clip_value_max=1e200)

    # get maps
    map_dg = poisson_sample(lam=poisson_lam, shape=(batch_size, n_pix, n_pix, n_zbins))
    map_dg = map_dg / tf.math.reduce_sum(map_dg, axis=(2,3), keepdims=True) * poisson_total
    map_dg = tf.math.round(map_dg)

    # tf.debugging.assert_all_finite(map_dg, message='found not finite values in map_dg')

    return map_dg




@tf.function()
def image_mean_subtract(X):
    """
    Subtract the mean, per image/zbin
    """

    X_double = tf.cast(X, tf.float64)
    X_double_m = X_double - tf.reduce_mean(X_double, axis=(2,3), keepdims=True)
    return tf.cast(X_double_m, tf.float32)


@tf.function()
def density_contrast(rho):
    """
    Convert to density contrast
    """

    rho_double = tf.cast(rho, tf.float64)
    rho_bar = tf.reduce_mean(rho_double, axis=(2,3),  keepdims=True)
    delta = (rho_double-rho_bar)/rho_bar
    return tf.cast(delta, tf.float32)


@tf.function()
def add_phase_noise(map_dg, r_stoch, n_pix, batch_size, n_fields, n_zbins):
    """
    Add random noise to phases
    """

    gen_rand = tf.random.get_global_generator()
    m_fft = tf.signal.rfft2d(tf.transpose(map_dg, perm=[0,1,4,2,3]), fft_length=(n_pix, n_pix)) # n_zbins, n_pix/2+1, npix, n_batch, n_fields, n_pix, n_pix, n_channels => n_batch, n_fields, n_channels, n_pix, n_pix
    m_fft_abs = tf.math.abs(m_fft)
    m_fft_phase = tf.math.angle(m_fft) 
    rand_phase = (1-r_stoch)**(2/3) * gen_rand.uniform(minval=-np.pi, maxval=np.pi, shape=(batch_size, n_fields, n_zbins, n_pix, n_pix//2+1))
    m_prime_fft_phase = m_fft_phase+rand_phase
    m_prime_fft = tf.complex(m_fft_abs*tf.math.cos(m_prime_fft_phase), m_fft_abs*tf.math.sin(m_prime_fft_phase))
    m_prime = tf.signal.irfft2d(m_prime_fft, fft_length=(n_pix, n_pix)) # n_zbins, n_pix, npix
    m_prime = tf.transpose(m_prime, perm=[0,1,3,4,2]) # n_batch, n_fields, n_channels, n_pix, n_pix -> n_batch, n_fields, n_pix, n_pix, n_channels
    # m_prime -= tf.math.reduce_mean(m_prime, axis=(2,3,4), keepdims=True)

    return m_prime

@tf.function()
def poisson_sample_clipped(lam_base, shape):

    lam_sum = tf.reduce_sum(lam_base, axis=(2,3), keepdims=True)
    lam_clip = tf.clip_by_value(lam_base, clip_value_min=0, clip_value_max=1e5)
    lam_clip_sum = tf.reduce_sum(lam_clip, axis=(2,3), keepdims=True)
    lam = lam_clip * lam_sum / lam_clip_sum # normalize to keep the number counts
    p = poisson_sample(lam=lam, shape=shape)
    return tf.math.round(p)

@tf.function()
def scale_kg(map_kg, n_gal_per_pix, sigma_e):
    """
    Transform kg map to span a reasonable numerical range
    """

    pixel_sig = sigma_e/tf.math.sqrt(n_gal_per_pix)
    a = 1./tf.math.sqrt(0.005**2 + pixel_sig**2)
    map_kg *= a

    map_kg = tf.where(tf.math.is_inf(map_kg), x=0., y=map_kg)
    map_kg = tf.where(tf.math.is_nan(map_kg), x=0., y=map_kg)

    return map_kg

@tf.function()
def scale_dg(map_dg, n_gal_per_pix):
    """
    Transform dg map to span a reasonable numerical range
    """

    a = 1./n_gal_per_pix
    b = -1.
    map_dg *= a
    map_dg += b
    map_dg = tf.where(tf.math.is_inf(map_dg), x=b, y=map_dg)
    map_dg = tf.where(tf.math.is_nan(map_dg), x=b, y=map_dg)

    return map_dg



















