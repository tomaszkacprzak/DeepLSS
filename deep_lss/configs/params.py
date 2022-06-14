import numpy as np
"""
All parameters of the Pipeline are defined here
"""

####################
# General Parameters
####################

n_per_cosmo = 12 # Number of simulations per cosmology
name_scheme = "cosmo_OmegaM_sigma8_num" # Nameing scheme of the cosmologies (see cosmo.par for options)
                                        # num is the number of the realisation (default is the end of string)

#########################
# Cosmological Parameters
#########################

# The cosmological parameters are read from a file
cosmo = "cosmo.par" # cosmological parameter file
boxsize = 500 # Boxsize in true Mpc
z_init = 50 # Initial Redshift

######################
# MUSIC ICS Parameters
######################

music_exec = "/users/jafluri/pkdgrav3/ohahn-music-afefabeea948/MUSIC" # path to MUSIC executable
music_level = 8 # The number of particles will be 2^level
music_out = "ics.dat" # Name of the output IC file


###################
# PKDGRAV parameter
###################

pkd_exec = "/users/jafluri/pkdgrav3/dpotter-pkdgrav3-c71a030e4360/build/pkdgrav3" # path to the PKDGRAV executable
pkd_run  = 60*45           # estimated runtime for one simulation
pkd_out_z = "output.red"   # File with the output redshifts
pkd_namespace = "KiDs450"  # Namespace used for all output files
pkd_steps = 500            # number of absolute timesteps
pkd_outint = 500           # output intervall in steps
pkd_cudaqueue = 64         # GPU CUDA queue length


################################
# Shells and lightcone Parameter
################################
shell_n = 1000 # Number of different shells per snapshot
shell_pix = 128 # Number of pixels for the shell/lightcone projection
shell_seed = None # Seed for the different cutouts (None Music seed[8] + shell is used)
shell_mem = 5000 # requested megabytes of RAM to generate the shells
shell_dur = "24:00" # Requested time to generate the shells

lc_origin = np.array([boxsize/2.0, boxsize/2.0, 0.0]) # Origin of the lightcone
lc_aperture = 5.0 # Aperture of the lightcone in deg
lc_mode = "costume" # Currently the only supported mode
# lc_nz_files = ["NZ_KiDs_meanDIR_0p1_0p3.txt", # Files to read out the n(z)
#                "NZ_KiDs_meanDIR_0p3_0p5.txt",
#                "NZ_KiDs_meanDIR_0p5_0p7.txt",
#                "NZ_KiDs_meanDIR_0p7_0p9.txt",
#                "NZ_KiDs_meanDIR_total.txt"]
# lc_names = ["0p1_0p3", "0p3_0p5", "0p5_0p7", "0p7_0p9", "total"] # Name extensions for the generated kappa maps

lc_names = ["zbin1", "zbin2", "zbin3", "zbin4", "ztot"] # Name extensions for the generated kappa maps
lc_nz_files = ["nz_DES_1.txt", "nz_DES_2.txt", "nz_DES_3.txt", "nz_DES_4.txt", "nz_DES_total.txt"]


lc_z_bounds = np.array([0.0, 3.45]) # Redshift boundaries for the lightcone
lc_mem = 5000 # Requsted Megabytes to project the lightcone
lc_dur = "4:00" # requested time to project the lightcone

################################
# TFrecords settings
################################
n_files_per_cosmo = 40
n_instances_per_file = 300
fiducial_cosmo_index=24
