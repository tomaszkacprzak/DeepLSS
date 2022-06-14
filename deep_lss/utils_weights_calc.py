from scipy.interpolate import interp1d
from scipy import integrate
from numpy import sqrt
import numpy as np, argparse, os


"""
Calculates the weights of the cosmologies of the grids and saves them
"""

speed_of_light_c = 299792.458  # Speed of light

def scale_kappa_weight(weight_kappa, H0, OmegaM, pix_area):

    return (3.0 * OmegaM / 2.0) * (H0 / speed_of_light_c) * weight_kappa * pix_area

def scale_dg_weight(weight_kappa, pix_area):

    return weight_kappa * pix_area


def scale_IA_weight(weight_IA, pix_area):

    return weight_IA * pix_area

def calculate_pixarea(params, z, delta_z, H0, OmegaL, OmegaM):

    # Calculate Pixelarea (assume all pixels have same area)
    # pix_area = (129600 * params.shell_pix ** 2 / (4 * np.pi ** 2 * params.lc_aperture ** 2))  # Fraction of the sphere

    # arXiv:1801.05745v1 Eq 2.7
    pix_area = (360**2/npi.pi) / (params.lc_aperture ** 2) # sphere part
    pix_area *= params.shell_pix ** 2 / (4 * np.pi ** 2)
    pix_area *= (H0 / speed_of_light_c) ** 2
    pix_area *= 1.0 / dimless_com_nz(0, (z + z + delta_z) / 2.0, H0, OmegaL, OmegaM) ** 2
    # Get number of paricels and adjust
    Nsim = (2 ** params.music_level) ** 3 # number of particles
    pix_area *= params.boxsize ** 3 / Nsim
    return pix_area


def shell_redshift(f, params):

    # Get redshift and delta_z
    f_1 = f.replace('shells_num_{}_'.format(params.shell_n), '')
    f_1 = f_1.replace('.npy', '')
    f_1 = f_1.split('_')
    try:
        z = float(f_1[0])
    except Exception as err:
        import ipdb; ipdb.set_trace()
        print(err)
        pass

    delta_z = float(f_1[1])
    return z, delta_z


def cosmo_rename(cosmo, num, params):
    """
    Renames the cosmology with the parameters cosmo to the naming scheme defined in params.py
    :param cosmo: Cosmological parameters ordered like the cosmo.par file
    :param num: Number of the iteration
    :return: the string of the cosmology and the hardcoded label
    """
    cosmo_str = params.name_scheme

    # replace the namespaces
    cosmo_str = cosmo_str.replace("OmegaM", "%.3f" %(cosmo[0]))
    cosmo_str = cosmo_str.replace("OmegaL", "%.3f" %(cosmo[1]))
    cosmo_str = cosmo_str.replace("w0", "%.3f" %(cosmo[2]))
    cosmo_str = cosmo_str.replace("wa", "%.3f" %(cosmo[3]))
    cosmo_str = cosmo_str.replace("Omegab", "%.3f" %(cosmo[4]))
    cosmo_str = cosmo_str.replace("hubble", "%.3f" %(cosmo[5]))
    cosmo_str = cosmo_str.replace("sigma8", "%.3f" %(cosmo[6]))
    cosmo_str = cosmo_str.replace("nspec", "%.3f" %(cosmo[7]))

    if "num" in cosmo_str:
        cosmo_str = cosmo_str.replace("num", str(num))
    else:
        cosmo_str += "_%i" %(num)

    # label is hard coded Om, sigma8
    label = np.array([cosmo[0], cosmo[6]])

    return cosmo_str, label


def F_NIA_model(z, H0, OmegaM):
    """
    Implementa F_NIA of the intrinsic alignment model
    :param z: redshift
    :param H0: Hubble parameter
    :param OmegaM: Matter density
    :return: F_NIA(z)
    """
    growth = lambda a: 1.0 / (a ** 3.0 * (OmegaM * a ** -3.0 + (1.0 - OmegaM)) ** 1.5)
    a = 1.0 / (1.0 + z)
    g = 5.0 * OmegaM / 2.0 * np.sqrt(OmegaM * a ** -3.0 +
                                           (1.0 - OmegaM)) * integrate.quad(growth, 0, a)[0]

    # Calculate the growth factor today
    g_norm = 5.0 * OmegaM / 2.0 * integrate.quad(growth, 0, 1)[0]

    # divide out a
    g = g / g_norm

    # Gravitational constant (km^2 Mpc Msun^-1 s^-2) units made to cancel with units of h such
    # that the result is the same units as C1
    G = 4.301e-9

    # critical density today = 3*params["h"]^2/(8piG)
    rho_c = 3 * (H0) ** 2 / (8 * np.pi * G)

    # Proportionality constant Msun^-1 Mpc^3
    C1 = 5e-14 / (H0/100.0) ** 2

    return -rho_c * C1 * OmegaM / g

def dimless_com_nz(zinit, zfinal, H0, OmegaL, OmegaM):
    """
    Calculates dimensionless comoving coordinate from redshifts zinit to zfinal.
    :param zinit: lower integration boundary
    :param zfinal: upper integration boundary
    :return: value for the dimensionless comoving coordinate
    """

    comoving = integrate.quad(lambda x: 1 / sqrt(OmegaM * (1 + x) ** 3 + OmegaL),
                              zinit, zfinal)
    return comoving[0]

def w_dirac(z, zs, dz, H0, OmegaL, OmegaM):
    """
    Calculates the slice-related weight with a Dirac-delta shaped redshift distribution at the source redshift zs.
    :param i: index related to the slice
    :param zs: source redshift
    :return: value of the weight for redshift-slice i and source redshift zs
    """
    f = integrate.quad(lambda x: 1 / sqrt(OmegaM * (1 + x) ** 3 + OmegaL) *
                       dimless_com_nz(0, x, H0, OmegaL, OmegaM) * dimless_com_nz(x, zs, H0, OmegaL, OmegaM) /
                       dimless_com_nz(0, zs, H0, OmegaL, OmegaM) * (1 + x), z, z + dz)[0]
    g = integrate.quad(lambda y: 1 / sqrt(OmegaM * (1 + y) ** 3 + OmegaL),
                       z, z + dz)[0]
    return f / g


def w_dg(z, nz, z_bounds, dz, points, H0, OmegaL, OmegaM):
    """
    Calculates the slice-related galaxy density contrast weight with a a given distribution of source redshifts n(z).
    # https://arxiv.org/abs/2007.05735 eqn 3.9
    :param i: index related to the redshift-slice
    :param nz: Distribution of source redshifts
    :return: value of the weight for redshift-slice i and source redshift distribution n(z)
    """
    
    # get break points for integration 
    # points = points[np.logical_and(z_bounds[0] < points, points < z_bounds[-1])]
    points = None # skip this for smooth distr

    def H(z):

        return H0 * sqrt(OmegaM * (1 + z) ** 3 + OmegaL)

    def g(z):
        return 1 / sqrt(OmegaM * (1 + z) ** 3 + OmegaL)

    def f(z):   

        bz = 1 # constant bias with redshift
        # return g(z) * H(z) * bz * nz(z) 
        print('WARNING - bug in W for dg, remove the H0 factor (it does not matter later though, so no rush..)')
        return H0 * bz * nz(z) 

    # integrate over n(z)
    w_shell = integrate.quad(f, z, z + dz)[0]

    # normalization
    norm = integrate.quad(g, z, z + dz)[0]

    return w_shell / norm

def w_nz(z, nz, z_bounds, dz, points, H0, OmegaL, OmegaM):
    """
    Calculates the slice-related lensing weight with a a given distribution of source redshifts n(z).
    :param i: index related to the redshift-slice
    :param nz: Distribution of source redshifts
    :return: value of the weight for redshift-slice i and source redshift distribution n(z)
    """
    def f(y, x):
        return 1 / sqrt(OmegaM * (1 + x) ** 3 + OmegaL) * (nz(y) *
                                                                dimless_com_nz(0, x, H0, OmegaL, OmegaM) *
                                                                dimless_com_nz(x, y, H0, OmegaL, OmegaM) /
                                                                dimless_com_nz(0, y, H0, OmegaL, OmegaM)) * (1 + x)

    # points = points[np.logical_and(z_bounds[0] < points, points < z_bounds[-1])]
    # quad_y = lambda x: integrate.quad(lambda y: f(y,x), x, z_bounds[-1], limit=1000,
    #                                   points=points[np.logical_and(x < points, points < z_bounds[-1])])[0]

    # skip points for smooth n(z)
    quad_y = lambda x: integrate.quad(lambda y: f(y,x), x, z_bounds[-1], limit=1000)[0]


    dbl = integrate.quad(quad_y, z, z + dz)[0]
    #dbl = integrate.dblquad(f, z, z + dz, lambda x: x, lambda x: z_bounds[-1])

    def g(y):
        return 1 / sqrt(OmegaM * (1 + y) ** 3 + OmegaL)

    norm1 = integrate.quad(g, z, z + dz)
    # norm2 = integrate.quad(nz, z_bounds[0], z_bounds[-1], points=points, limit=1000)
    norm2 = integrate.quad(nz, z_bounds[0], z_bounds[-1], points=None, limit=1000)

    return dbl / (norm1[0]*norm2[0])


def w_IA(z, nz, z_bounds, dz, points, H0, OmegaL, OmegaM):
    """
    Calculates the slice-related weight for the NIA model  with a a given distribution of source redshifts n(z).
    :param i: index related to the redshift-slice
    :param nz: Distribution of source redshifts
    :return: value of the weight for redshift-slice i and source redshift distribution n(z)
    """
    c = 299792.458

    def f(x):
        return H0 / c * (F_NIA_model(x, H0=H0, OmegaM=OmegaM) * nz(x))

    # def f(x):
    #    return 1 / sqrt(params["Omega0"] * (1 + x) ** 3 + params["OmegaLambda"]) * (F_NIA_model(x, params)*nz(x))

    points = points[np.logical_and(z_bounds[0] < points, points < z_bounds[-1])]
        
    # switch off points for now
    dbl = integrate.quad(f, z, z + dz)[0]

    def g(y):
        return 1 / sqrt(OmegaM * (1 + y) ** 3 + OmegaL)

    norm1 = integrate.quad(g, z, z + dz)
    # norm2 = integrate.quad(nz, z_bounds[0], z_bounds[-1], points=points, limit=100)
    norm2 = integrate.quad(nz, z_bounds[0], z_bounds[-1], points=None, limit=100)
    # norm2 = integrate.quad(lambda x: (c/params["H0"])*nz(x)*g(x), z_bounds[0], z_bounds[-1], points=points, limit=100)

    return dbl / (norm1[0] * norm2[0])


