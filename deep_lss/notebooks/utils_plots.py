import os, sys, shutil, stat, logging, subprocess, shlex, collections, datetime, numpy as np, pickle, importlib, h5py, itertools, seaborn
import matplotlib
import pylab as plt
from collections import OrderedDict
from deep_lss.filenames import *
from deep_lss import utils_io, utils_arrays, utils_logging
from tqdm import trange
plt.style.use('seaborn-paper')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

LOGGER = utils_logging.get_logger(__file__)



colors_algos = {'cnn' : 'deeppink', 
                'psd' : 'dodgerblue'}
labels_algos = {'psd':'Power spectrum', 'cnn':'Deep learning'}

labels = {'area2d': r'\rm{FoM} $\Omega_m/\sigma_8$',
          'S8': r'$S_8$',
          'Omega_m': r'$\Omega_m$',
          'sigma_8': r'$\sigma_8$',
          'A_IA': r'$A_{\mathrm{IA}}$',
          'b_gal': r'$b_g$',
          'r_stoch': r'$r_g$',
          'eta_A_IA': r'$\eta_{A_{\mathrm{IA}}}$',
          'eta_b_gal': r'$\eta_{b_g}$',
          'eta_b_gal': r'$\eta_{b_g}$',
          'b2_gal': r'$b_{g,2}$',
          'eta_b2_gal': r'$\eta_{b_{g,2}}$'}

theta_lims = {'Omega_m': [0.2, 0.4],
              'sigma_8': [0.7, 0.9], 
              'A_IA': [-3,6], 
              'b_gal': [1.2, 1.8], 
              'S8': [0.65, 0.9], 
              'r_stoch': [0.6, 0.8],
              'eta_A_IA': [-4, 6], 
              'eta_b_gal': [0.2, 0.7],
              'b2_gal': [-3, 1],
              'eta_b2_gal': [-2, 2]}

theta_priors = {'Omega_m': [0.15, 0.45],
                'sigma_8': [0.5, 1.2],
                'A_IA': [-6, 6] ,
                'b_gal': [0.5, 2.5] ,
                'r_stoch': [0.4, 1] ,
                'eta_A_IA': [-5, 7],
                'eta_b_gal': [-2, 2]}

field_side_arcmin = 5*60
field_area_arcmin2 = field_side_arcmin**2
img_size = 128
pix_area = field_area_arcmin2/128**2
n_gal_per_arcmin2 = 2.5
n_gal_per_pix = n_gal_per_arcmin2*pix_area

def add_S8(samples):

    samples = utils_arrays.add_cols(samples, ['S8'])
    samples['S8'] = samples['sigma_8']*(samples['Omega_m']/0.3)**(0.5)
    return samples

def plot_example_maps(path, vlim=[-0.02, 0.2], id_sample=0, smoothing_arcmin=1.17, n_maps=2):


    smoothing_arcmin = { 0 : [4.778, 3.509, 2.758, 2.526],
                         1 : [9.839, 7.398, 5.991, 5.567],
                         2 : [4.778, 3.509, 2.758, 2.526] }
    smoothing_pix =  lambda x: np.array(x)/(field_side_arcmin/img_size)
    
    def trans_kg(x):

        return x-np.mean(x)

    def trans_dg(x):

        return (x - np.mean(x))/np.mean(x)

    trans = {0: trans_kg, 
             1: trans_dg,
             2: trans_kg}

    from scipy.ndimage import gaussian_filter

    n_z_bins=4
    # n_maps = 2
    maps = ['kg', 'dg', 'IA']
    signal_to_mapid = {'kg':0, 'IA':1, 'dg': 2}
    z_bins_order = [0, 1, 2, 3]
    colormaps = {'kg': plt.cm.plasma, 
                 'dg': plt.cm.viridis, 
                 'IA': plt.cm.cividis_r}


    colormaps = {'kg': seaborn.color_palette("magma", as_cmap=True),
                 'dg': seaborn.color_palette("viridis", as_cmap=True),
                 'IA': seaborn.color_palette("plasma", as_cmap=True)}

    

    vlims = {'kg': [-0.015, 0.025], 'dg': [-0.5, 0.5], 'IA': [-0.005, 0.002]}
    signal_labels = {'kg': r'weak lensing', 
                     'dg': r'galaxy clustering', 
                     'IA': r'intrinsic alignment'}
    fontsize_labels = 20
    fontsize_colorbar = 14
    fontsize_tags = 16
    zbin_labels = {'kg': r'$\kappa_g \ \ z_{\rm{bin}}$',
                   'dg': r' $\delta_g \ \  z_{\rm{bin}}$',     
                   'IA': r' $\kappa_{\rm{IA}} \ \  z_{\rm{bin}}$'}     

    nx, ny = n_maps, n_z_bins; fig, ax = plt.subplots(nx, ny, figsize=(ny * 4, nx * 4), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    
    with h5py.File(path, 'r') as f:
        for i in range(n_maps):
            
            sig = smoothing_pix(smoothing_arcmin[i])
            
            for j in range(n_z_bins):

                a = ax[i,j]
                m = maps[i]

                map_id = signal_to_mapid[m]
                vlim = vlims[m]

                img = np.array(f[str(id_sample)][:,:,map_id,j])

                img = trans[i](img)

                img = gaussian_filter(img, sigma=sig[j])

                # cb=a.pcolormesh(img, cmap=colormaps[m], rasterized=True)
                cb=a.pcolormesh(img, vmin=vlims[m][0], vmax=vlims[m][1], cmap=colormaps[m], rasterized=True)
                a.set(xticklabels=[], yticklabels=[], yticks=[], xticks=[])
                a.text(x=120, y=112, s=zbin_labels[m] + f'={j+1}', fontsize=fontsize_tags, rotation='horizontal', backgroundcolor='white', ha="right", bbox=dict(facecolor='white', alpha=0.8, edgecolor=None))



                if j == 0:
                    a.set_ylabel(signal_labels[m], fontsize=fontsize_labels, rotation='vertical', labelpad=15)
            
            cax = plt.axes([0.74, (n_maps-i-1)*0.4+0.2, 0.2, 0.2])
            cax.set_visible(False)
            cbar = fig.colorbar(cb, ax=cax, aspect=10)
            cbar.ax.tick_params(labelsize=fontsize_colorbar) 
            cbar.ax.set_yticklabels(["{: 4.2f}".format(i) for i in cbar.get_ticks()]) # set ticks of your format

    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    return fig




def temp_remove_outliers(samples, sig=2.5):

    from scipy.stats import sigmaclip

    in_lims = lambda x, lims: (x>=lims[0]) & (x<=lims[1])

    select = np.ones(len(samples), dtype=np.bool)
    for c in samples.dtype.names:
       _, lo, hi =sigmaclip(samples[c], low=sig, high=sig) 

       select &= in_lims(samples[c], [lo, hi])
    return samples[select]



def plot_constraints_triangle(results_prefix, id_sample=None, field='validation', plottype='density_image', probes=['cp', 'dg', 'kp'], algos=['psd', 'cnn']):

    from trianglechain.utils_plots import contour_cl
    from trianglechain.TriangleChain import TriangleChain
    from deep_lss.configs import params

    def add_prob(samples):
    
        samples = utils_arrays.add_cols(samples, ['prob'])
        select = np.isfinite(samples['log_prob'])
        samples['prob'][select] = np.exp(samples['log_prob'][select]-np.max(samples['log_prob'][select]))
        samples['prob'][~select]=0
        samples['prob'] /= np.sum(samples['prob'])
        return samples

    def add_legend(coords=[0.46, 0.45, 0.1, 0.1]):

        from matplotlib.patches import Patch
        handles = [Patch(facecolor=colors_algos[alg], label=labels_algos[alg])  for alg in algos]
        lax = plt.axes(coords)
        lax.axis('off')
        lax.legend(handles=handles, fontsize=fontsize_legend, loc='center')

    def add_title(a, x, y, probe):

        a.text(x=x,
               y=y,
               s=probes_conf[probe]['tag'],
               fontsize=fontsize_tags,
               # backgroundcolor=probes_conf['kg']['tag_color'],
               ha="center",
               transform=a.transAxes)

    def load_chain(probe, alg):


        if id_sample is not None:

            with h5py.File(f'{results_prefix}/constraints/constraints_merged.h5', 'r') as f:
                samples = np.array(f[f'{field}/base_{probe}_{algos_runs[alg]}/samples_{id_sample:05d}'])
                y_true = np.array(f[f'{field}/base_{probe}_{algos_runs[alg]}/y_true_{id_sample:05d}'])

        else:

            d = utils_io.read_from_pickle(f'{results_prefix}/constraints/constraints_stats.pkl')
            samples = d['chain_stack'][f'{probe}_{algos_runs[alg]}']
            with h5py.File(f'{results_prefix}/constraints/constraints_merged.h5', 'r') as f:
                y_true = np.array(f[f'{field}/base_{probe}_{algos_runs[alg]}/y_true_00000'])


        samples = add_S8(samples)
        y_true = add_S8(y_true)

        return samples, y_true



    field='comparison'
    de_kwargs = {}
    de_kwargs.setdefault('n_points', 100)
    de_kwargs.setdefault('levels', [0.68, 0.95])
    de_kwargs.setdefault('n_levels_check', 2000)
    de_kwargs.setdefault('smoothing_sigma', 1)
    de_kwargs['levels'].sort()


    fontsize_label = 24
    fontsize_tags = 24
    fontsize_legend = 22
    fontsize_ticklabels = 16
    tag_pos = [0.95, 0.85]
    hspace = 0.3
    n_de_grid = 100
    star_size = 100
    star_marker = 'o'

    grid_kwargs = {}
    grid_kwargs['tickformat'] = '{: 2.2f}'
    grid_kwargs['fontsize_ticklabels'] = fontsize_ticklabels

    labels_kwargs = {}
    labels_kwargs['fontsize'] = fontsize_label

    theta_cosmo = utils_io.load_theta_cosmo('./', params)


    probes_conf = {'kg': {'par2': 'A_IA', 'tag': r'weak lensing $\kappa_g$', 'par2_label': r'$A_{\rm{IA}}$', 'tag_color':'plum'},
                   'dg': {'par2': 'b_gal', 'tag': r'galaxy clustering $\delta_g$', 'par2_label': r'$b_{g}$', 'tag_color':'skyblue'},
                   'cp': {'tag': r'combined probes $\kappa_g + \delta_g$', 'tag_color':'pink'}}

    list_figs = []

    algos_runs = {'cnn': 'cnn',
                  'psd': 'psd'}


    # lensing + clustering
    probe = 'cp'
    if probe in probes:

        theta_lims = {'Omega_m': [0.2, 0.4],
                      'sigma_8': [0.675, 0.9], 
                      'A_IA': [-3,6], 
                      'b_gal': [1.2, 1.8], 
                      'S8': [0.65, 0.9], 
                      'r_stoch': [0.6, 0.75],
                      'eta_A_IA': [-5, 7], 
                      'eta_b_gal': [0.3, 0.7]}
        
        theta = ['Omega_m', 'sigma_8', 'A_IA', 'b_gal', 'r_stoch', 'eta_A_IA', 'eta_b_gal']
        from trianglechain.TriangleChain import TriangleChain
        tri = TriangleChain(ranges=theta_lims,
                            n_bins=n_de_grid,
                            labels=labels,
                            plot_histograms_1D=True,
                            size=4,
                            de_kwargs=de_kwargs,
                            grid_kwargs=grid_kwargs, 
                            labels_kwargs=labels_kwargs)

        for alg in algos:

            samples, y_true = load_chain(probe, alg)
            print(samples['Omega_m'].shape)
            fom = get_figure_of_merit(s1=samples['Omega_m'], s2=samples['sigma_8'])
            print(alg, fom)

            # Omega_m - sigma_8 panel
            samples = samples[theta]
            y_true = y_true[theta]
            # y_true_full = y_true_full[theta]


            # tri.scatter(y_true_full, color='grey')
            tri.contour_cl(samples, color=colors_algos[alg], fill=True, grid=True)
            tri.scatter(y_true, color='black', scatter_kwargs={'zorder':10, 's':star_size, 'marker':star_marker})

        add_legend(coords=[0.23, 0.72, 0.2, 0.2])
        add_title(a=tri.ax[1,0], x=1.95, y=1.75, probe='cp')
        list_figs.append(tri.fig)


    fontsize_label = 28
    fontsize_tags = 30
    fontsize_legend = 30
    fontsize_ticklabels = 20
    grid_kwargs['fontsize_ticklabels'] = fontsize_ticklabels

    labels_kwargs = {}
    labels_kwargs['fontsize'] = fontsize_label


    # lensing
    de_kwargs['smoothing_sigma'] = 2 # smooth kg more given that many parameters are unconstrained

    probe = 'kg'
    if probe in probes:

        theta_lims = {'Omega_m': [0.15, 0.45],
                      'sigma_8': [0.5, 1.1], 
                      'A_IA': [-6,6], 
                      'eta_A_IA': [-5, 6]}

        theta = ['Omega_m', 'sigma_8', 'A_IA', 'eta_A_IA']
        tri = TriangleChain(ranges=theta_lims,
                            n_bins=n_de_grid,
                            labels=labels,
                            plot_histograms_1D=True,
                            de_kwargs=de_kwargs,
                            grid_kwargs=grid_kwargs, 
                            labels_kwargs=labels_kwargs)

        for alg in algos:
        # for alg in ['psd']:

            samples, y_true = load_chain(probe, alg)
            
            # Omega_m - sigma_8 panel
            samples = samples[theta]
            y_true = y_true[theta]

            tri.contour_cl(samples, color=colors_algos[alg], fill=True, grid=True)
            tri.scatter(y_true, color='black', scatter_kwargs={'zorder':10, 's':star_size, 'marker':star_marker})

        add_legend(coords=[0.39, 0.68, 0.2, 0.2])
        add_title(a=tri.ax[1,0], x=1.65, y=1.8, probe='kg')

        
        list_figs.append(tri.fig)
        

    # clustering


    probe = 'dg'
    de_kwargs['smoothing_sigma'] = 1
    de_kwargs['n_points'] = 100

    if probe in probes:

        theta_lims = {'Omega_m': [0.15, 0.45],
                      'sigma_8': [0.5, 1.1], 
                      'b_gal': [1.2, 1.8], 
                      'r_stoch': [0.6, 0.8],
                      'eta_b_gal': [0.2, 0.7]}
    
        theta = ['Omega_m', 'sigma_8', 'b_gal', 'r_stoch', 'eta_b_gal']
        tri = TriangleChain(ranges=theta_lims,
                            n_bins=n_de_grid,
                            labels=labels,
                            plot_histograms_1D=True,
                            de_kwargs=de_kwargs,
                            grid_kwargs=grid_kwargs,
                            labels_kwargs=labels_kwargs)

        for alg in algos:

            samples, y_true = load_chain(probe, alg)

            # Omega_m - sigma_8 panel
            samples = samples[theta]
            y_true = y_true[theta]

            # tri.scatter(y_true_full, color='grey')
            tri.contour_cl(samples, color=colors_algos[alg], fill=True)
            tri.scatter(y_true, color='black', scatter_kwargs={'zorder':10, 's':star_size, 'marker':star_marker})

        add_legend(coords=[0.32, 0.7, 0.2, 0.2])
        add_title(a=tri.ax[1,0], x=1.8, y=1.8, probe='dg')

        list_figs.append(tri.fig)


    return list_figs





def plot_compare_IA_S8(results_prefix, id_sample=None):

    def load_chain(probe, alg):


        if id_sample is not None:

            with h5py.File(f'{results_prefix}/constraints/constraints_merged.h5', 'r') as f:
                samples = np.array(f[f'{field}/base_{probe}_{algos_runs[alg]}/samples_{id_sample:05d}'])
                y_true = np.array(f[f'{field}/base_{probe}_{algos_runs[alg]}/y_true_{id_sample:05d}'])

        else:

            d = utils_io.read_from_pickle(f'{results_prefix}/constraints/constraints_stats.pkl')
            samples = d['chain_stack'][f'{probe}_{algos_runs[alg]}']
            with h5py.File(f'{results_prefix}/constraints/constraints_merged.h5', 'r') as f:
                y_true = np.array(f[f'{field}/base_{probe}_{algos_runs[alg]}/y_true_00000'])

        samples = add_S8(samples)
        y_true = add_S8(y_true)

        return samples, y_true

    params=['A_IA', 'S8'] 
    probes=['kg', 'cp']
    field='comparison'

    from trianglechain.utils_plots import contour_cl

    nx, ny = 1, 2; fig, ax = plt.subplots(nx, ny, figsize=(ny * 6, nx * 4), squeeze=False); axc=ax[0,0]; axl=ax[0,:];

    algos_runs = {'cnn': 'cnn',
                  'psd': 'psd'}


    de_kwargs = {}
    de_kwargs.setdefault('n_points', 100)
    de_kwargs.setdefault('levels', [0.68, 0.95])
    de_kwargs.setdefault('n_levels_check', 2000)
    de_kwargs.setdefault('smoothing_sigma', 1)
    de_kwargs['levels'].sort()


    labels_probes = {'kg': r'weak lensing $\kappa_g$',
                     'dg': r'galaxy clustering $\delta_g$',
                     'cp': r'combined probes $\kappa_g + \delta_g$'}

    fontsize_label = 20
    fontsize_legend = 18
    star_size = 100
    fontsize_title = 16
    fontsize_ticks = 14
    fontsize_tags = 18
    ylabelpad = -2

    theta_lims = {'Omega_m': [0.15, 0.45],
                 'sigma_8': [0.5, 1.2], 
                 'A_IA': [-4,6], 
                 'b_gal': [1.2, 1.8], 
                 'S8': [0.6, 0.9], 
                 'r_stoch': [0.6, 0.8],
                 'eta_A_IA': [-4, 6], 
                 'eta_b_gal': [0.2, 0.7]}
    
    algos = ['psd', 'cnn']
    for j, probe in enumerate(probes):
        
        for i, alg in enumerate(algos):


            a = ax[0,j]

            samples, y_true = load_chain(probe, alg)

            samples = samples[params]

            contour_cl(axc=a, data=samples, color=colors_algos[alg], fill=True, ranges=theta_lims, columns=params, i=1, j=0, de_kwargs=de_kwargs, alpha=1)
            a.grid(True, linestyle='--', zorder=0)
            a.set_axisbelow(True)
            a.set_xlabel(labels[params[0]], fontsize=fontsize_label)
            a.text(x=0.05, y=0.87, s=labels_probes[probe], fontsize=fontsize_tags, rotation='horizontal', ha="left", transform=a.transAxes)


        a.scatter(y_true[params[0]], y_true[params[1]], color='black', zorder=10, s=star_size, marker='o')
        a.tick_params(direction='in', length=0)
        a.set_yticks([0.675, 0.75 , 0.825])
        a.set_xticks([-1.5,  1. ,  3.5])


    # ax[0,1].set_yticklabels([])

    ax[0,0].set_ylabel(labels[params[1]], fontsize=fontsize_label)#, labelpad=ylabelpad)
    ax[0,1].set_ylabel(labels[params[1]], fontsize=fontsize_label)#, labelpad=ylabelpad)
    for a in ax.ravel():
        plt.setp(a.get_yticklabels(), fontsize=fontsize_ticks)
        plt.setp(a.get_xticklabels(), fontsize=fontsize_ticks)


    seaborn.color_palette("Paired")
    # fig.subplots_adjust(wspace=0.04, hspace=0.05)
    fig.subplots_adjust(wspace=0.3, hspace=0.35)



    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors_algos[alg], label=labels_algos[alg])  for alg in algos]
    # lax = plt.axes([0.64, 0.13, 0.1, 0.1])

    # lax = plt.axes([0.35, 0.7, 0.1, 0.1])
    # lax.axis('off')
    # legend = lax.legend(handles=handles, fontsize=fontsize_legend, loc='center', frameon=True, facecolor='white', edgecolor=None, framealpha=1)
    legend = ax[0,1].legend(handles=handles, fontsize=fontsize_legend, loc='lower left', frameon=False, facecolor='white', edgecolor=None, framealpha=1)

    return fig



def plot_compare_nonlin_bias(results_prefix, id_sample=None):

    def load_chain(probe, alg):

        if id_sample is not None:

            with h5py.File(f'{results_prefix}/constraints/constraints_merged.h5', 'r') as f:
                samples = np.array(f[f'{field}/{probe}_{algos_runs[alg]}/samples_{id_sample:05d}'])
                y_true = np.array(f[f'{field}/{probe}_{algos_runs[alg]}/y_true_{id_sample:05d}'])

        else:

            d = utils_io.read_from_pickle(f'{results_prefix}/constraints/constraints_stats.pkl')
            samples = d['chain_stack'][f'{probe}_{algos_runs[alg]}']
            with h5py.File(f'{results_prefix}/constraints/constraints_merged.h5', 'r') as f:
                y_true = np.array(f[f'{field}/{probe}_{algos_runs[alg]}/y_true_00000'])

        samples = add_S8(samples)
        y_true = add_S8(y_true)

        return samples, y_true

    params=[['b_gal', 'Omega_m'], ['b_gal', 'b2_gal'], ['b_gal', 'eta_b_gal'], ['b2_gal', 'eta_b2_gal']] 
    probes=['cp']
    field='comparison'

    from trianglechain.utils_plots import contour_cl

    nx, ny = 2, 2; fig, ax = plt.subplots(nx, ny, figsize=(ny * 6, nx * 4), squeeze=False); axc=ax[0,0]; axl=ax[0,:];

    algos_runs = {'cnn': 'cnn',
                  'psd': 'psd'}


    de_kwargs = {}
    de_kwargs.setdefault('n_points', 100)
    de_kwargs.setdefault('levels', [0.68, 0.95])
    de_kwargs.setdefault('n_levels_check', 2000)
    de_kwargs.setdefault('smoothing_sigma', 1)
    de_kwargs['levels'].sort()


    labels_probes = {'kg': r'weak lensing $\kappa_g$',
                     'dg': r'galaxy clustering $\delta_g$',
                     'cp': r'combined probes $\kappa_g + \delta_g$'}

    fontsize_label = 20
    fontsize_legend = 18
    star_size = 100
    fontsize_title = 16
    fontsize_ticks = 14
    fontsize_tags = 18
    theta_lims = {'Omega_m': [0.2, 0.4],
                 'sigma_8': [0.5, 1.2], 
                 'A_IA': [-4,6], 
                 'b_gal': [1.2, 1.8], 
                 'S8': [0.6, 0.9], 
                 'r_stoch': [0.6, 0.8],
                 'eta_A_IA': [-4, 6], 
                 'eta_b_gal': [0.2, 0.8],
                 'b2_gal': [-2, 0],
                 'eta_b2_gal': [-2, 2]}

    theta_ticks = {'b_gal': [1.35, 1.5, 1.65],
                   'eta_b_gal': [0.35,  0.5 ,  0.65],
                   'Omega_m': [0.25, 0.3, 0.35],
                   'b2_gal': [-1.5, -1, -0.5],
                   'eta_b2_gal': [-1, 0, 1], 
                   'r_stoch': [0.4, 0.7, 1]}

    algos = ['psd', 'cnn']
    for j, probe in enumerate(probes):
        
        for i, alg in enumerate(algos):

            k = 0
            for p, pars in enumerate(params):

                a = ax.transpose().ravel()[k]

                samples, y_true = load_chain(probe, alg)
                samples = samples[pars]

                contour_cl(axc=a, data=samples, color=colors_algos[alg], fill=True, ranges=theta_lims, columns=pars, i=1, j=0, de_kwargs=de_kwargs, alpha=1)
                a.grid(True, linestyle='--', zorder=0)
                a.set_axisbelow(True)
                a.set_xlabel(labels[pars[0]], fontsize=fontsize_label)
                a.set_ylabel(labels[pars[1]], fontsize=fontsize_label)
                # ax[0,0].set_xticklabels([])
                # a.set_title(labels_algos[alg], fontsize=fontsize_title)

                a.text(x=0.05, y=0.87, s=labels_probes[probe], fontsize=fontsize_tags, rotation='horizontal', ha="left", transform=a.transAxes)
                a.set_xticks(theta_ticks[pars[0]])
                a.set_yticks(theta_ticks[pars[1]])
                plt.setp(a.get_yticklabels(), fontsize=fontsize_ticks)
                plt.setp(a.get_xticklabels(), fontsize=fontsize_ticks)
                a.tick_params(direction='in', length=0)


                a.scatter(y_true[pars[0]], y_true[pars[1]], color='black', zorder=10, s=star_size, marker='o')

                k+=1


    # ax[0,0].set_xticklabels([])
    # plt.setp(ax[1,0].get_yticklabels(), fontsize=fontsize_ticks)
    # plt.setp(ax[1,0].get_xticklabels(), fontsize=fontsize_ticks)


    seaborn.color_palette("Paired")
    fig.subplots_adjust(wspace=0.3, hspace=0.35)


    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors_algos[alg], label=labels_algos[alg])  for alg in algos]
    lax = plt.axes([0.46, 0.89, 0.1, 0.1])

    # lax = plt.axes([0.35, 0.7, 0.1, 0.1])
    lax.axis('off')
    # legend = lax.legend(handles=handles, fontsize=fontsize_legend, loc='center', frameon=True, facecolor='white', edgecolor=None, framealpha=1, ncol=2)
    legend = ax[1,1].legend(handles=handles, fontsize=fontsize_legend, loc='lower left', frameon=False, facecolor='white', edgecolor=None, framealpha=1)

    return fig
















def plot_compare_smoothing_scales(results_prefix, recompute_foms=False):

    def get_contour_area(s1, s2, reso=200):
            
        sig = np.min([np.std(s1), np.std(s2)])/10
        lvl = 0.68
        bins1 = np.linspace(0, 0.6, reso)
        bins2 = np.linspace(0.6, 1.5, reso)
        bins_area = (bins1[-1]-bins1[0]) * (bins2[-1]-bins2[0])
        h = np.histogram2d(s1, s2, bins=(bins1, bins2))[0]
        h /= h.sum()
        l = get_confidence_levels(h, levels=[lvl])[0]
        area = np.count_nonzero(h>l)/len(h.ravel()) * bins_area
        return area

    def get_confidence_levels(de, levels, n_levels_check=1000):

        lvl_max = 0.99
        levels_check = np.linspace(0, np.amax(de)*lvl_max, n_levels_check)
        frac_levels = np.zeros_like(levels_check)

        for il, vl in enumerate(levels_check):
            pixels_above_level = de > vl
            frac_levels[il] = np.sum(pixels_above_level * de)

        levels_contour = [levels_check[np.argmin(np.fabs(frac_levels - level))] for level in levels][::-1]
        # print('levels_contour', levels_contour/np.amax(de)/lvl_max)
        # if np.any(levels_contour==levels_check[-1]):
            # print('contour hitting the boundary level {}'.format(str(levels_contour/np.amax(de)/lvl_max)))
        return levels_contour



    field= 'comparison'
    probes = ['kg', 'dg', 'cp']
    scales_tags = ['sig1p17', 'sig2p34', 'sig3p51', 'sig4p68', 'sig5p85']
    scales_sigmas = [1.17, 2.34, 3.51, 4.68, 5.85]
    n_chains = 100

    tags_full = ['_'.join(t) for t in itertools.product(probes, algos, scales_tags)]

    if recompute_foms:
        
        res = OrderedDict()
        for t in tags_full:
            res[t] = np.zeros(n_chains)
        
        for pro in probes:
            
            with h5py.File(f'{results_prefix}/{pro}/constraints/constraints_merged.h5', 'r') as f:

                for alg, tag, j in LOGGER.progressbar(list(itertools.product(algos, scales_tags, range(n_chains))), desc=f'FOMs for probe {pro}'):

                    samples = np.array(f[f'{field}/{alg}_{tag}/samples_{j:05d}'])
                    # samples = temp_remove_outliers(samples)
                    y_true = np.array(f[f'{field}/{alg}_{tag}/y_true_{j:05d}'])
                    y_true_full = np.array(f[f'{field}/y_true_full'])
                    res[f'{pro}_{alg}_{tag}'][j] = get_contour_area(s1=samples['Omega_m'], s2=samples['sigma_8'], reso=100)

        utils_io.write_to_pickle('foms.pkl', res)

    res = utils_io.read_from_pickle('foms.pkl')


    # plot specs
    ylim = [0, None]
    xlim = [1,6]
    fontsize_label = 12
    fontsize_title = 14
    fontsize_legend = 12
    fontsize_ticklabels = 10
    labels_probes = {'kg': r'weak lensing $\kappa_g$',
                     'dg': r'galaxy clustering $\delta_g$',
                     'cp': r'combined probes $\kappa_g + \delta_g$'}

    
    nx, ny = 1, len(probes); fig, ax = plt.subplots(nx, ny, figsize=(ny * 5, nx * 3), squeeze=False); axc=ax[0,0]; axl=ax[0,:];0
    for i, pro in enumerate(probes):
        for j, alg in enumerate(algos):

            fom_med = np.array([1./np.median(res[f'{pro}_{alg}_{tag}']) for tag in scales_tags])
            fom_min = np.array([1./np.percentile(res[f'{pro}_{alg}_{tag}'], 16) for tag in scales_tags])
            fom_max = np.array([1./np.percentile(res[f'{pro}_{alg}_{tag}'], 84) for tag in scales_tags])

            axl[i].fill_between(scales_sigmas, fom_min, fom_max, label=labels_algos[alg], color=colors_algos[alg])
        
        axl[i].set(ylim=ylim, xlim=xlim)
        axl[i].set_title(labels_probes[pro], fontsize=fontsize_title)
        axl[i].grid(True, linestyle='--', zorder=0)
        axl[i].set_axisbelow(True)
        axl[i].tick_params(axis='both', labelsize=fontsize_ticklabels)
        axl[i].set_xlabel('smoothing scale [arcmin]', fontsize=fontsize_label)
        if i == 0:
            axl[i].set_ylabel('figure of merit FoM\n' + r'$\leftarrow$ worse      better $\rightarrow$', fontsize=fontsize_label)

        # if i > 0:
            # axl[i].set_yticklabels([])


        if i == len(probes)-1:
            axl[i].legend(fontsize=fontsize_legend)

    fig.subplots_adjust(wspace=0.2)

    return fig


def plot_compare_advantage(results_prefix, recompute_foms=False):

    # plot specs

    field= 'comparison'
    probes = ['kg', 'dg', 'cp']
    scales_tags = ['sig1p17', 'sig2p34', 'sig3p51', 'sig4p68', 'sig5p85']
    scales_sigmas = [1.17, 2.34, 3.51, 4.68, 5.85]
    ylim = [0, 5]
    xlim = [1,6]
    fontsize_label = 12
    fontsize_title = 14
    fontsize_legend = 12
    fontsize_ticklabels = 10
    labels_probes = {'kg': r'weak lensing $\kappa_g$',
                     'dg': r'galaxy clustering $\delta_g$',
                     'cp': r'combined probes $\kappa_g + \delta_g$'}

    colors_probes = {}
    # colors_probes['kg'] = 'orangered'
    # colors_probes['dg'] = 'limegreen'
    # colors_probes['cp'] = 'darkorchid'

    colors_probes['kg'] = seaborn.color_palette("Paired")[1]
    colors_probes['dg'] = seaborn.color_palette("Paired")[3]
    colors_probes['cp'] = seaborn.color_palette("Paired")[5]

    res = utils_io.read_from_pickle('foms.pkl')

    nx, ny = 1, 1; fig, ax = plt.subplots(nx, ny, figsize=(ny * 5, nx * 3), squeeze=False); axc=ax[0,0]; axl=ax[0,:];0
    for i, pro in enumerate(probes):

        fom_ratio_min = np.array([np.percentile(res[f'{pro}_psd_{tag}'] / res[f'{pro}_cnn_{tag}'], 16) for tag in scales_tags])
        fom_ratio_max = np.array([np.percentile(res[f'{pro}_psd_{tag}'] / res[f'{pro}_cnn_{tag}'], 84) for tag in scales_tags])

        axc.fill_between(scales_sigmas, fom_ratio_min, fom_ratio_max, color=colors_probes[pro], label=labels_probes[pro])

    axc.legend(fontsize=fontsize_legend)
    axc.grid(True, linestyle='--', zorder=0)
    axc.set_axisbelow(True)
    axc.set(ylim=ylim, xlim=xlim)
    axc.set_xlabel('smoothing scale [arcmin]', fontsize=fontsize_label)
    axc.set_ylabel(r'$\rm{FoM}^{\rm{CNN}}/\rm{FoM}^{\rm{PSD}}$' +'\n' + r'$\leftarrow$ worse      better $\rightarrow$', fontsize=fontsize_label,  labelpad=15)
    axc.tick_params(axis='both', labelsize=fontsize_ticklabels)

    return fig




def plot_gain_summary(results_prefix, results_prefix_nonlin,  n_chains=200):



    theta_labels = {}
    theta_labels['base_cp'] = ['Omega_m', 'sigma_8', 'S8', 'area2d', 'A_IA', 'b_gal', 'r_stoch', 'eta_A_IA', 'eta_b_gal']
    theta_labels['base_dg'] = ['Omega_m', 'sigma_8', 'S8', 'area2d', 'b_gal', 'r_stoch', 'eta_b_gal']
    theta_labels['base_kg'] = ['Omega_m', 'sigma_8', 'S8', 'area2d', 'A_IA', 'eta_A_IA']
    theta_labels['nonlin_cp'] = ['Omega_m', 'sigma_8', 'S8', 'area2d', 'A_IA', 'b_gal', 'r_stoch', 'eta_A_IA', 'eta_b_gal', 'b2_gal', 'eta_b2_gal']
    theta_labels['nonlin_dg'] = ['Omega_m', 'sigma_8', 'S8', 'area2d', 'b_gal', 'r_stoch', 'eta_b_gal', 'b2_gal', 'eta_b2_gal']
    theta_labels['nonlin_kg'] = ['Omega_m', 'sigma_8', 'S8', 'area2d', 'A_IA', 'eta_A_IA']

    f_base = utils_io.read_from_pickle(f'{results_prefix}/constraints/constraints_stats.pkl')
    f_nonlin = utils_io.read_from_pickle(f'{results_prefix_nonlin}/constraints/constraints_stats.pkl')
    

    probes = ['kg', 'dg', 'dg', 'cp', 'cp']
    column_breaks = [0,2]
    probes_cases = ['base_kg',  'base_dg', 'nonlin_dg', 'base_cp', 'nonlin_cp']
    theta_all = ['Omega_m', 'sigma_8', 'S8', 'area2d', 'A_IA', 'b_gal', 'r_stoch', 'eta_A_IA', 'eta_b_gal', 'b2_gal', 'eta_b2_gal']
    f_case = lambda case: f_base if 'base' in case else f_nonlin

    labels_macros = {'Omega_m':    '\omatter',
                     'sigma_8':    '\sigeight',
                     'S8':         '\Seight',
                     'A_IA':       '\AIA',
                     'b_gal':      '\\biasg',
                     'r_stoch':    '\\rstoch',
                     'eta_A_IA':   '\etaIA',
                     'eta_b_gal':  '\etab',
                     'b2_gal':     '\\biasnl',
                     'eta_b2_gal': '\etabnl',
                     'area2d':     r'\rm{FoM} \omatter -- \sigeight'}

    offset_str = lambda s: '{:>40s}'.format(s)
    lines_theta = {t: '{:<40s} & &'.format(labels_macros[t]) for t in theta_all}

    for p, (probe, case) in enumerate(zip(probes, probes_cases)):

        theta_lab = theta_labels[case]
        n_theta = len(theta_lab)
        if 'base' in case:
            model_cnn = f'base_{probe}_cnn'
            model_psd = f'base_{probe}_psd'
        else:
            model_cnn = f'{probe}_cnn'
            model_psd = f'{probe}_psd'

        models = [model_psd, model_cnn]
        f = f_case(case)

        model_std = {}
        for i, m in enumerate(models):
        
            model_std[m] = {theta: np.zeros(n_chains) for theta in theta_lab}
        
            for j in range(n_chains):
        
                for t, theta in enumerate(theta_lab):
                    model_std[m][theta][j] = f['chain_stats'][f'comparison'][m][j][f'std_{theta}' if theta != 'area2d' else 'area2d']

        for i, theta in enumerate(theta_all):

            if theta in model_std[model_psd].keys():

                std_psd = model_std[model_psd][theta]
                std_cnn = model_std[model_cnn][theta]
                s = r'{:2.1f}$\times$ '.format(np.median(std_psd/std_cnn))
                lines_theta[theta] += offset_str(s)
            else:
                lines_theta[theta] += offset_str(' --  ')

            if p < len(probes)-1:
                lines_theta[theta] += offset_str(' & ')

                if p in column_breaks:
                    lines_theta[theta] += ' & '

    for theta, line in lines_theta.items():
        lines_theta[theta] += ' \\\\'

    table_rows = ""
    for theta, line in lines_theta.items():
        table_rows += line + '\n'
            
                         # \cmidrule(lr){3-3}
    table_start = r""" \begin{tabularx}{0.8\textwidth}{lXrXrrXrr}
                         \toprule
                           & & \multicolumn{1}{c}{\textbf{Weak Lensing}} &  & \multicolumn{2}{c}{\textbf{Galaxy Clustering}} &  & \multicolumn{2}{c}{\textbf{Combined Probes}} \\
                         \cmidrule(lr){5-6} 
                         \cmidrule(lr){8-9} 
                              & & \ & & large scales & small scales & & large scales &small scales\\
                             \midrule
                   """

    table_end = r""" \bottomrule
                     \end{tabularx}
                    """

    return table_start + '\n' + table_rows + '\n' + table_end


def plot_sensitivity(results, id_survey, id_field, id_z_bin=0, index=0):

    theta_labels = {}
    theta_labels['base_cp'] = ['Omega_m', 'sigma_8', 'A_IA', 'b_gal', 'r_stoch', 'eta_A_IA', 'eta_b_gal']
    theta_labels['base_dg'] = ['Omega_m', 'sigma_8', 'b_gal', 'r_stoch', 'eta_b_gal']
    theta_labels['base_kg'] = ['Omega_m', 'sigma_8', 'A_IA', 'eta_A_IA']
    theta_labels['noia'] = ['S8', 'Omega_m', 'sigma_8']
    theta_labels['nonlin_cp'] = ['Omega_m', 'sigma_8', 'A_IA', 'b_gal', 'r_stoch', 'eta_A_IA', 'eta_b_gal', 'b2_gal', 'eta_b2_gal']
    theta_labels['nonlin_dg'] = ['Omega_m', 'sigma_8', 'b_gal', 'r_stoch', 'eta_b_gal', 'b2_gal', 'eta_b2_gal']
    theta_labels['nonlin_kg'] = ['Omega_m', 'sigma_8', 'A_IA', 'eta_A_IA']

    sens_maps_cnn = utils_io.read_from_pickle(f'{results}/predictions/sensitivity_maps_index{index:02d}.pkl')
    # sens_maps_psd = utils_io.read_from_pickle(f'{results}/predictions/sensitivity_maps.pkl')
    case = 'base_cp'
    model = 'cp_cnn'

    def get_X(sens_maps, model, noise=False):
                
        t = 'X_noisy' if noise else 'X_nonoise'
        X_kg = sens_maps[model][t][id_survey, :64, :64, 0:4]
        X_dg = sens_maps[model]['X_noisy'][id_survey, :64, :64, 4:8]
        return X_kg, X_dg

    def get_S(sens_maps, model):
        S = sens_maps[model]['dX_do'][:,:n_params,...]
        S_single = S[id_survey,id_param,id_field,...]
        S_kg = S_single[...,0:4]
        S_dg = S_single[...,4:8]
        return S_kg, S_dg
    
    thetas = theta_labels[case]
    n_params = len(theta_labels[case])
    id_field = 0
    # id_param = thetas.index(theta)
    # X_kg, X_dg = get_X(sens_maps_cnn, model=model, noise=True)
    # S_kg_cnn, S_dg_cnn = get_S(sens_maps_cnn, model=model)
    # S_kg_psd, S_dg_psd = get_S(sens_maps_psd, model='cp_psd')

    fontsize_tags = 20
    fontsize_title = 20
    color_thresh = 99.9
    vmin, vmax = -1.5e-4, 1.5e-4

    def plot_sensitivity_for_zbin(id_z_bin_kg, id_z_bin_dg):

        def add_plot(img, pos, cmap, label, vmin='auto', vmax='auto', smooth_pix=None):

            if smooth_pix is not None:
                from scipy.ndimage.filters import gaussian_filter
                img = gaussian_filter(img, sigma=smooth_pix, mode='constant')

            if type(vmin) is str:
                if vmin == 'auto':
                    vmin = np.percentile(img, 0)
            if type(vmax) is str:
                if vmax == 'auto':
                    vmax = np.percentile(img, 100)
            
            a = ax[pos]
            cb = a.pcolormesh(img, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
            a.text(x=3, y=58, s=label, fontsize=fontsize_tags, rotation='horizontal', backgroundcolor='white', ha="left", bbox=dict(facecolor='white', alpha=0.8, edgecolor=None))

        nx, ny = 2, 2; fig, ax = plt.subplots(nx, ny, figsize=(ny * 6, nx * 6), squeeze=False); axc=ax[0,0]; axl=ax[0,:];

        add_plot(X_kg[:,:,id_z_bin_kg], smooth_pix=1, vmin='auto', vmax='auto', pos=(0,0), cmap='plasma',  label=r'map $\kappa_g, \ z_{\mathrm{bin}}='+str(id_z_bin_kg+1)+'$')
        add_plot(X_dg[:,:,id_z_bin_dg], vmin='auto', vmax='auto', pos=(1,0), cmap='viridis', label=r'map $\delta_g, \ z_{\mathrm{bin}}='+str(id_z_bin_dg+1)+'$')

        add_plot(S_kg_cnn[:,:,id_z_bin_kg], vmin=-vlim_kg, vmax=vlim_kg, pos=(0,1), cmap='RdBu_r', label=r'sensitivity $\partial$' + labels[theta] + r'/$\partial \kappa_g,  \ z_{\mathrm{bin}}='+str(id_z_bin_kg+1)+'$')
        add_plot(S_dg_cnn[:,:,id_z_bin_dg], vmin=-vlim_dg, vmax=vlim_dg, pos=(1,1), cmap='RdBu_r', label=r'sensitivity $\partial$' + labels[theta] + r'/$\partial \delta_g,  \ z_{\mathrm{bin}}='+str(id_z_bin_dg+1)+'$')

        # add_plot(S_kg_psd[:,:,id_z_bin], pos=(0,2), cmap='RdBu', label=r'sensitivity PSD $\partial A_{\rm{IA}}/\partial \kappa_g $')
        # add_plot(S_dg_psd[:,:,id_z_bin], pos=(1,2), cmap='RdBu', label=r'sensitivity PSD $\partial A_{\rm{IA}}/\partial \delta_g $')

        # remove ticks
        for a in ax.ravel():
            a.set_xticks(np.linspace(0,64,7))
            a.set_yticks(np.linspace(0,64,7))
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.tick_params(length=0)
            a.grid(True, ls='--', color='black')

        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        return fig

    theta = 'A_IA'
    vlim_kg = 1e-4
    vlim_dg = 1e-4
    id_param = thetas.index(theta)
    X_kg, X_dg = get_X(sens_maps_cnn, model=model, noise=True)
    S_kg_cnn, S_dg_cnn = get_S(sens_maps_cnn, model=model)
    fig1 = plot_sensitivity_for_zbin(id_z_bin_kg=0, id_z_bin_dg=0)
    fig1.suptitle(f'sensitivity for {labels[theta]}', y=0.91, fontsize=fontsize_title)

    theta = 'sigma_8'
    vlim_kg = 3e-5
    vlim_dg = 3e-5
    id_param = thetas.index(theta)
    X_kg, X_dg = get_X(sens_maps_cnn, model=model, noise=True)
    S_kg_cnn, S_dg_cnn = get_S(sens_maps_cnn, model=model)
    fig2 = plot_sensitivity_for_zbin(id_z_bin_kg=3, id_z_bin_dg=1)
    fig2.suptitle(f'sensitivity for {labels[theta]}', y=0.91, fontsize=fontsize_title)

    return fig1, fig2


def plot_reshift_bins(path_nzs='../redshift_distributions'):

    def bin_mean(z, nz):
        return np.sum(nz*z)/np.sum(nz)

    def load_bins(paths):
        list_nz = []
        for p in paths:
            list_nz += [np.loadtxt(os.path.join(path_nzs,p))]
        return list_nz

    n_z_bins = 4
    paths2 = [f'nz_DES_{i}.txt' for i in range(1, n_z_bins+1)]
    nz2 = load_bins(paths2)

    fontsize_title = 12
    import seaborn
    colors = seaborn.color_palette('colorblind', 4)

    nx, ny = 1, 1; fig, ax = plt.subplots(nx, ny, figsize=(ny * 4*1.25, nx * 3*1.25), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    for i in range(n_z_bins):
        z_mean = bin_mean(nz2[i][:,0], nz2[i][:,1])
        tag = r'$z_{\rm{bin}}$=' + f'{i+1}' + r' $\langle z \rangle$={:2.2f}'.format(z_mean)
        print(tag)
        axc.plot(nz2[i][:,0], nz2[i][:,1], label=tag, c=colors[i], lw=4)
    axc.set(xlim=[0,2], ylim=[0, None])
    axc.tick_params(labelsize=fontsize_title)
    axc.set_yticks([])
    axc.legend(fontsize=fontsize_title)
    axc.set_xlabel('redshift z', fontsize=fontsize_title)
    axc.tick_params(direction='in', length=5)

    return fig



def get_figure_of_merit(s1, s2):
    

    lvl = 0.68
    bins1 = np.linspace(0, 0.6, 400)
    bins2 = np.linspace(0.4, 1.5, 400)
    bins_area = (bins1[-1]-bins1[0]) * (bins2[-1]-bins2[0])
    h = np.histogram2d(s1, s2, bins=(bins1, bins2))[0]
    h /= h.sum()
    l = get_confidence_levels(h, levels=[lvl])[0]
    area = np.count_nonzero(h>l)/len(h.ravel()) * bins_area
    return area




def get_confidence_levels(de, levels, n_levels_check=1000, n_bins=50):

    from scipy.optimize import bisect

    levels_contour = []
    for cl in levels:
        cl_level = bisect(f=lambda l: np.sum(np.array(de > l) * de)-cl, a=0, b=1)
        levels_contour.append(cl_level)

    return levels_contour



def plot_stochasticity_relation(path='./cosmo_0.291_0.783_0/kappa_maps.h5'):

    def density_contrast(m):
    
        m_bar = np.mean(m)
        m = (m-m_bar)/m_bar
        return m

    def get_mprime(X, b_gal, r_sto):
        
        m_fft = np.fft.rfft2(X*b_gal, axes=(0, 1))
        m_fft_abs = np.abs(m_fft)
        m_fft_phase = np.angle(m_fft) 
        rand_phase = (1-r_sto)**(2/3) * np.random.uniform(low=-np.pi, high=np.pi, size=m_fft.shape)
        m_prime_fft_phase = m_fft_phase+rand_phase
        m_prime_fft = m_fft_abs*np.exp(1j*m_prime_fft_phase)
        m_prime = np.fft.irfft2(m_prime_fft, axes=(0,1))
        m_prime -= np.mean(m_prime)
        return m_prime
        
    n_z_bins=4
    map_id=2
    n_maps=36
    fontsize_xlabel = 12
    fontsize_ylabel = 18
    
    grid_r = np.linspace(0, 1, 20)
    cross = np.zeros((len(grid_r), n_z_bins, n_maps))
    with h5py.File(path, 'r') as f:
        for j in range(n_z_bins):
            for i, r in enumerate(grid_r):
                for k in range(n_maps):
                    m = np.array(f[f'{k}'][:,:,map_id,j])
                    m = density_contrast(m)
                    mprime = get_mprime(m, b_gal=1, r_sto=r)
                    cross[i, j, k] = np.mean(m*mprime)/np.mean(m*m)
    cross = cross.mean(axis=-1)

    nx, ny = 1, 1; fig, ax = plt.subplots(nx, ny, figsize=(ny * 4*1.25, nx * 3*1.25), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    for i in range(n_z_bins):
        axc.plot(grid_r, cross[:,i], color='k')

    axc.plot(grid_r, grid_r, color='k', ls='--')

    axc.set(ylim=[0,1], xlim=[0,1])
    axc.set_xlabel('galaxy stochasticity ' + labels['r_stoch'], fontsize=fontsize_xlabel)
    axc.set_ylabel(r'$\frac{\langle \delta_m \delta_m^r \rangle}{\langle \delta_m \delta_m \rangle}$', fontsize=fontsize_ylabel)
    axc.tick_params(direction='in', length=5)
    fig.tight_layout()
    axc.tick_params(labelsize=fontsize_xlabel)


    return fig





def plot_likemodel_check(path):

    def in_lims(p, lims):

        return (lims[0]<=p) & (p<=lims[1])

    def in_box(theta):

        inside = np.ones(len(theta), dtype=bool)
        for t in theta.dtype.names:
            if t in theta_priors:
                inside &= in_lims(theta[t], theta_priors[t])  
        return inside

    def split_by_true(y_true, y_samp, y_pred):
            
        hashing = utils_arrays.rec_to_arr(y_true).sum(axis=1)
        uv = np.unique(hashing)
        y_true_split, y_samp_split, y_pred_split = [], [], []
        for v in uv:
            select = hashing == v
            y_true_split.append(y_true[select])
            y_samp_split.append(y_samp[select])
            y_pred_split.append(y_pred[select])       
        
        return y_true_split, y_samp_split, y_pred_split
    
        

    def load_lssnet_predictions(file_pred, squeeze=False, flatten=False, models_filter=None):

        import h5py
        samp = False
        nets_y_pred = {}
        nets_y_true = {}
        nets_y_samp = {}
        with h5py.File(file_pred, 'r') as f:
            for n in f.keys():
                nets_y_pred[n] = np.array(f[n]['y_pred'])
                nets_y_true[n] = np.array(f[n]['y_true'])
                if 'y_samp' in f[n].keys():
                    nets_y_samp[n] = np.array(f[n]['y_samp'])
                    samp = True

        if flatten:
            for n in nets_y_true.keys():
                nets_y_true[n] = nets_y_true[n].reshape(-1, nets_y_true[n].shape[-1])
            for n in nets_y_pred.keys():
                nets_y_pred[n] = nets_y_pred[n].reshape(-1, nets_y_pred[n].shape[-1])
                if samp:
                    for n in nets_y_samp.keys():
                        nets_y_samp[n] = nets_y_samp[n].reshape(-1, nets_y_samp[n].shape[-1])

        if squeeze:
            nets_y_true[n] = nets_y_true[n].squeeze()
            nets_y_pred[n] = nets_y_pred[n].squeeze()
            if samp:
                nets_y_samp[n] = nets_y_samp[n].reshape(-1, nets_y_samp[n].shape[-1])

        netnames = []
        for netname in nets_y_true.keys():
            if models_filter is not None:
                if models_filter in netname:
                    netnames+=[netname]
            else:
                netnames+=[netname]
        nets_y_true = {n:nets_y_true[n] for n in netnames}
        nets_y_pred = {n:nets_y_pred[n] for n in netnames}    

        if samp:
            nets_y_samp = {n:nets_y_samp[n] for n in netnames}    
        
        if samp:
            return nets_y_true, nets_y_pred, nets_y_samp
        else:
            return nets_y_true, nets_y_pred


    def plot_mean_difference_summary(path):

        from scipy.stats import median_absolute_deviation as mad
        import glob
        fnames = np.sort(glob.glob(os.path.join(path, 'predictions', f'predictions__likenet_verif__*.h5')))
        
        nx, ny = 2, 2;  fig, ax = plt.subplots(nx, ny, figsize=(ny * 4, nx * 3), squeeze=False); axc=ax[0,0]; axl=ax[0,:];

        list_all_diff_mean, list_all_diff_stdv = [], []

        for j, fname in enumerate(fnames):
            
            y_true, y_pred, y_samp = load_lssnet_predictions(fname, flatten=False)
            model = list(y_true.keys())[0]
            y_true, y_pred, y_samp = y_true[model], y_pred[model], y_samp[model]

            select = in_box(y_true)
            y_true = y_true[select]
            y_pred = y_pred[select]
            y_samp = y_samp[select]
            print(f'{model} selected in prior {np.count_nonzero(select)} {len(select)}')

            y_true, y_samp, y_pred = split_by_true(y_true, y_samp, y_pred)
            n_cosmo, n_theta = len(y_true), len(y_true[0].dtype.names)

            list_diff_mean, list_diff_stdv = [], []
            for i, t in enumerate(y_true[0].dtype.names):
                mean_pred = np.array([np.mean(y_pred[j][t]) for j in range(n_cosmo)])
                mean_samp = np.array([np.mean(y_samp[j][t]) for j in range(n_cosmo)])
                stdv_pred = np.array([np.std(y_pred[j][t]) for j in range(n_cosmo)])
                stdv_samp = np.array([np.std(y_samp[j][t]) for j in range(n_cosmo)])

                frac_diff = (mean_samp-mean_pred)/stdv_pred
                list_diff_mean.append(frac_diff)

                frac_diff = (stdv_samp-stdv_pred)/stdv_pred
                list_diff_stdv.append(frac_diff)

            diff_mean = np.concatenate(list_diff_mean)
            diff_stdv = np.concatenate(list_diff_stdv)
            bins = np.linspace(-1,1,60)

            list_all_diff_mean.append(diff_mean)
            list_all_diff_stdv.append(diff_stdv)

            print(f'mean {path} {np.percentile(diff_mean, 16):2.2f} {np.percentile(diff_mean, 84):2.2f}')
            print(f'stdv {path} {np.percentile(diff_stdv, 16):2.2f} {np.percentile(diff_stdv, 84):2.2f}')

            a = get_ax(model, ax)[0]
            a.hist(diff_mean, bins=bins, histtype='step', lw=histlw, label=model_labels[model], density=True)
            a.axvline(0, c='k', ls='-')
            a.axvspan(-0.3, 0.3, alpha=0.1, color='grey', zorder=0)
            plt.setp(a.get_xticklabels(), fontsize=fontsize_ticks)
            a.set_yticks([])
            a.set_xticks(xticks_val)
            a.legend(loc='upper left', frameon=False, fontsize=fontsize_legend)

            a = get_ax(model, ax)[1]
            a.hist(diff_stdv, bins=bins, histtype='step', lw=histlw, label=model_labels[model], density=True)
            a.axvline(0, c='k', ls='-')
            a.axvspan(-0.2, 0.2, alpha=0.1, color='grey', zorder=0)
            plt.setp(a.get_xticklabels(), fontsize=fontsize_ticks)
            a.set_yticks([])
            a.set_xticks(xticks_val)
            a.legend(loc='upper left', frameon=False, fontsize=fontsize_legend)

        all_diff_mean = np.concatenate(list_all_diff_mean)
        all_diff_stdv = np.concatenate(list_all_diff_stdv)
        print(f'all {np.percentile(all_diff_mean, 16):2.2f} {np.percentile(all_diff_mean, 84):2.2f}')
        print(f'all {np.percentile(all_diff_stdv, 16):2.2f} {np.percentile(all_diff_stdv, 84):2.2f}')


        for a in ax[0,:]:
            a.set_xlabel(r'$\Delta_\mu$', fontsize=fontsize_label)
        for a in ax[1,:]:
            a.set_xlabel(r'$\Delta_\sigma$', fontsize=fontsize_label)

        for a in ax.ravel():
            a.set_xlim(min(bins), max(bins))

        for a in get_ax('cnn', ax):
            a.text(0.9, 0.8, s='CNN', transform=a.transAxes, fontsize=fontsize_legend, ha='right')
        for a in get_ax('psd', ax):
            a.text(0.9, 0.8, s='PSD-NN', transform=a.transAxes, fontsize=fontsize_legend, ha='right')

        fig.subplots_adjust(wspace=wspace, hspace=hspace)
        # fig.suptitle('Fractional difference of values of predicted summaries', fontsize=fontsize_title)
        return fig


    def get_ax(model, ax):

        if 'psd' in model:
            a = ax[:,0]
        elif 'cnn' in model:
            a = ax[:,1]
        return a

    model_labels = {'base_cp_cnn': r'$\kappa_g + \delta_g$',
                    'base_kg_cnn': r'$\kappa_g$',
                    'base_dg_cnn': r'$\delta_g$',
                    'base_cp_psd': r'$\kappa_g + \delta_g$',
                    'base_kg_psd': r'$\kappa_g$',
                    'base_dg_psd': r'$\delta_g$'}
    wspace = 0.05
    hspace = 0.4
    histlw = 2
    fontsize_tag = 14
    fontsize_label = 14
    fontsize_ticks = 12
    fontsize_title = 18
    fontsize_legend = 12

    xticks_val = [-0.5, 0, 0.5]
    fig1 = plot_mean_difference_summary(path)


    return fig1


def plot_psd_opt(path):
    """
    First save loss series from tensorboard
    """


    def load_tflog(fname):
            
        import glob
        print(fname, glob.glob(fname))
        fname_ = glob.glob(fname)[0]
        log = np.loadtxt(fname_, delimiter=',', skiprows=1)
        return log

    def smoothing(x):
        return np.convolve(x, np.ones(n_smooth)/n_smooth, mode='same')

    def add_plot(x, y, **kw):
        n = len(x)-n_smooth
        axc.plot(x[:n], smoothing(y)[:n], **kw)
    

    n_opt = 6
    n_smooth=20
    logs = {f'opt{i}' : load_tflog(f'./{path}/lssnet_logs_case{i+1}*.csv') for i in range(1,n_opt)}
    logs_cp_cnn = load_tflog(f'./{path}/lssnet_logs_base_cp_cnn.csv')
    logs_cp_psd = load_tflog(f'./{path}/lssnet_logs_base_cp_psd.csv')

    nx, ny = 1, 1; fig, ax = plt.subplots(nx, ny, figsize=(ny * 4.5, nx * 3.5), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    for k, v in logs.items():
        n = len(v)-n_smooth
        label = 'alternative CP-PSD-NN models' if 'opt1' in k else None
        add_plot(v[:,1], v[:,2], c='k', label=label, alpha=0.5)

    add_plot(logs_cp_cnn[:,1], logs_cp_cnn[:,2], c='deeppink', lw=3, label='selected CP-CNN model')
    add_plot(logs_cp_psd[:,1], logs_cp_psd[:,2], c='dodgerblue', lw=3, label='selected CP-PSD-NN model')
    axc.set_xlim(10000,900000)
    axc.set_ylim(-50,-25)
    axc.legend()
    axc.grid(True, linestyle='--', zorder=0)
    axc.set_xlabel('optimization step')
    axc.set_ylabel('loss')
    axc.set_xticks([1e5, 3e5, 5e5, 7e5])

    from matplotlib import ticker
    f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
    axc.xaxis.set_major_formatter(ticker.FuncFormatter(g))

    return fig



def measure_bias_error_from_desy3():
    
    def bias_evol(z, z0, b_gal, eta):
        f = b_gal*((1+z)/(1+z0))**(eta)
        return f

    def get_linear_bias_params_desy3():
        
        # arXiv:2012.08566v2 table 2
        z_meas = [0.343, 0.521, 0.742, 0.964]
        # arXiv:2105.13549v1 table V
        # Fid. Linear Bias 1.49+0.10 1.69+0.11 1.91+0.12 1.79+0.11
        b_meas = [1.49, 1.69, 1.91, 1.79]
        b_err = [0.1, 0.11, 0.12, 0.11]
        return np.array(z_meas), np.array(b_meas), np.array(b_err)

    def get_nonliner_bias_params_desy3():
        
        # arXiv:2012.08566v2 table 2
        z_meas = [0.343, 0.521, 0.742, 0.964]
        # arXiv:2105.13549v1 table V
        # Fid. Linear Bias 1.49+0.10 1.69+0.11 1.91+0.12 1.79+0.11
        b_meas = [1.44, 1.60, 1.85, 1.74]
        b2_meas = [0.12, 0.06, -0.06, 0.40]

        b_err = [0.08, 0.1, 0.11, 0.11]
        b2_err = [0.15, 0.35, 0.30, 0.32]
        
        return np.array(z_meas), np.array(b_meas), np.array(b_err), np.array(b2_meas), np.array(b2_err)


    def log_prob_lin(x, b_meas):
        
        b_model = bias_evol(z_meas, z0=0.7, b_gal=x[0], eta=x[1])
        chi2 = -0.5*np.sum((b_meas-b_model)**2/b_err**2)
        return chi2


    def get_2d_prob(prior0, prior1, b_meas):
        
        import chaospy
        distr = chaospy.J(chaospy.Uniform(*prior0), chaospy.Uniform(*prior1))
        grid_lin = distr.sample(size=int(1e5), rule='halton').T
        log_prob = np.array([log_prob_lin(x, b_meas) for x in grid_lin])
        prob = np.exp(log_prob - np.max(log_prob))
        prob = prob/np.sum(prob)
        
        bins0 = np.linspace(*prior0, 200)
        bins0_centers = bins_centers(bins0)
        bins1 = np.linspace(*prior1, 100)
        bins1_centers = bins_centers(bins1)
        pj, _, _ = np.histogram2d(x=grid_lin[:,0], y=grid_lin[:,1], bins=(bins0, bins1), weights=prob)
        p0, _ = np.histogram(grid_lin[:,0], bins=bins0, weights=prob)
        p1, _ = np.histogram(grid_lin[:,1], bins=bins1, weights=prob)
        p0 /= np.sum(p0)
        p1 /= np.sum(p1)
        
        return p0, p1, bins0_centers, bins1_centers

    bins_centers = lambda b: (b[:-1]+b[1:])/2.
    vec_std = lambda px, x: np.sqrt(np.sum( (x-np.mean(x))**2 * px))

    prior_b_gal = [1.5, 2] 
    prior_eta_b_gal = [-0.5, 1.5]
    prior_b2_gal = [-0.5, 0.5]
    prior_eta_b2_gal =[-10, 50]

    ## linear bias

    z_meas, b_meas, b_err=get_linear_bias_params_desy3()

    p0, p1, x0, x1 = get_2d_prob(prior_b_gal, prior_eta_b_gal, b_meas)

    print(f'linear bias b_g {vec_std(p0, x0):2.2f}')
    print(f'linear bias eta_b_g {vec_std(p1, x1):2.2f}')

    ## non-linear bias

    z_meas, b_meas, b_err, b2_meas, b2_err =get_nonliner_bias_params_desy3()

    p0, p1, x0, x1 = get_2d_prob(prior_b2_gal, prior_eta_b2_gal, b2_meas)
    print(f'non-linear bias b2_g {vec_std(p0, x0):2.2f}')
    print(f'non-linear bias eta_b2_g {vec_std(p1, x1):2.2f}')



