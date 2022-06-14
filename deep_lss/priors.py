import h5py, numpy as np, tensorflow as tf, itertools, warnings
from deep_lss import utils_logging, utils_arrays
from scipy.spatial import Delaunay

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

def convert_sigma8_to_S8(y):
    y_ = y.copy()
    y_['sigma_8'] = y['sigma_8']*(y['Omega_m']/0.3)**0.5
    return y_

def convert_S8_to_sigma8(y):
    y_ = y.copy()
    y_['sigma_8'] = y['sigma_8']/(y['Omega_m']/0.3)**0.5
    return y_



def in_lims(p, lims):

    return (lims[1]<=p) & (p<=lims[2])

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    # https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def unit_to_lims(a, lims):


    if type(lims) is list:
        a *= (lims[2]-lims[1])
        a += lims[1]
    else:
        a[:] = lims

    return a


class ThetaPriors():

    def __init__(self, ctx):
    
        self.theta_cosmo = ctx.theta_cosmo
        self.theta_full = ['Omega_m', 'sigma_8', 'A_IA', 'b_gal', 'r_stoch', 'alpha0_spoiss', 'alpha1_spoiss', 'eta_A_IA', 'eta_b_gal', 'b2_gal', 'eta_b2_gal'] # all parameters that can be varied in this model
        self.theta_full_dynamic = ['A_IA', 'b_gal', 'r_stoch', 'alpha0_spoiss', 'alpha1_spoiss', 'eta_A_IA', 'eta_b_gal', 'b2_gal', 'eta_b2_gal']
        self.theta_targets = self.theta_full
        self.theta_targets_cosmo = ['Omega_m', 'sigma_8']
        self.theta_cosmo_delaunay =  Delaunay(utils_arrays.rec_to_arr(self.theta_cosmo, cols=self.theta_targets_cosmo))

        self.prior_lims = ctx.priors
        # limits on cosmo parameters
        self.fid_cosmo_id = 24
        for p in self.theta_targets_cosmo:
            # self.prior_lims[p] = [self.theta_cosmo[p][self.fid_cosmo_id], np.min(self.theta_cosmo[p]), np.max(self.theta_cosmo[p])]
            self.prior_lims[p] = [self.theta_cosmo[p][self.fid_cosmo_id], self.prior_lims[p][1], self.prior_lims[p][2]]

        self.param_variable = {p: type(v) is list for p,v in self.prior_lims.items()}


        self.n_astro_params = len(self.theta_full_dynamic)
        self.n_grid = ctx.analysis_settings['n_grid_theta_astro']
        self.n_drawn = 0

    def __call__(self, seed=None, fix_fiducial=False):

        samples = self.sample(n_samples=self.n_grid, seed=seed)

        if fix_fiducial:
            
            for i, p in enumerate(self.theta_full_dynamic):
                samples[p][self.fid_cosmo_id::len(self.theta_cosmo)] = self.prior_lims[p][0]

        return samples

    def sample(self, n_samples, seed=None, rule='latin', rec=True, reset=False):

        if reset:
            self.n_drawn=0


        if rule == 'latin':
            from scipy.stats.qmc import LatinHypercube
            samples = LatinHypercube(d=self.n_astro_params, seed=seed).random(n_samples).astype(np.float32)
        
        elif rule == 'sobol':
            from sobol_seq import i4_sobol_generate
            samples = i4_sobol_generate(dim_num=self.n_astro_params, n=n_samples, skip=self.n_drawn).astype(np.float32)
            self.n_drawn += n_samples

        elif rule == 'uniform':
            samples = np.random.uniform(size=(n_samples, self.n_astro_params)).astype(np.float32)

        else:
            raise Exception(f'unknown rule {rule}')

        # scale to prior range
        for i, p in enumerate(self.theta_full_dynamic):
            unit_to_lims(samples[:,i], self.prior_lims[p])

        # convert to records
        if rec:
            samples = utils_arrays.arr_to_rec(samples, cols=self.theta_full_dynamic)

        LOGGER.debug(f'drew n={samples.shape[0]} LHC samples with seed={seed} first={samples[0]}')

        return samples

    def in_prior(self, theta):

        par_cosmo = np.array([theta[p] for p in self.theta_targets_cosmo]).T

        inside = self.in_hull(par_cosmo) & self.in_box(theta)

        return inside

    def in_hull(self, p):

        return in_hull(p, hull=self.theta_cosmo_delaunay)

    def in_box(self, theta):

        inside = np.ones(len(theta), dtype=bool)
        for t in self.theta_full:
            if (t in theta.dtype.names) and self.param_variable[t]:
                inside &= in_lims(theta[t], self.prior_lims[t])  
        return inside


    def load_theta_grid(self, rec=True):

        priors_theta_all = np.load(self.prior_grid)
        LOGGER.info(f'loaded {self.prior_grid} with {len(priors_theta_all)} gridpoints')
        priors_theta = priors_theta_all[self.theta_targets]

        if not rec:
            priors_theta = utils_arrays.rec_to_arr(rec=priors_theta, cols=self.theta_targets, dtype=np.float32, strict=False)
    
        return priors_theta

    def fix_fiducial(self, samples, index):

        for par in self.theta_full_dynamic:

            samples[par][index] = np.atleast_1d(self.prior_lims[par])[0]

    def get_fiducial(self):

        theta = self.sample(1, rule='uniform', rec=True)
        self.fix_fiducial(theta, index=0)
        return theta






