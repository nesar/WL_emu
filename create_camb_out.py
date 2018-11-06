import numpy as np
import subprocess as sp
from astropy.cosmology import FlatLambdaCDM
import pylab as pl
import time
from scipy.interpolate import interp1d
import pade_funcs as pade
from cython_funcs import c_sum_pk_2d

#----------------------------------------------------------------------
# Define the cosmology and constants used in this code.
#
vc = 2.998e5 #(km/s), speed of light
H0 = 71.0 #km/s / Mpc, Hubble constant
h = 0.71
Om0= 0.264 # Omega_M
ncosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

pk_dir = "./des_output_pk/" # the directory to save *_pk.dat
#----------------------------------------------------------------------
# calculate P(k,z) in $nbins$ redshift bins between zmin and zmax, and
# save the results in pk_dir. please see "./readme.txt" to learn
# how to run emu.exe to calculate P(k,z)
#
def main(zmin,zmax,nbins):
    dzl = (zmax-zmin)/nbins
    zl_array = np.linspace(zmin,zmax-dzl,nbins)+dzl/2.0 
    for i in xrange(nbins):
        filename = pk_dir+str(i)+"_"+str(zl_array[i])+"_"+str(dzl)+"_pk.dat"
        
        cmd = "./emu.exe "+filename +" 0.02258  0.1330824 0.963 71 -1.0 0.8 "+str(zl_array[i])
        sp.call(cmd,shell=True)

    return 0
