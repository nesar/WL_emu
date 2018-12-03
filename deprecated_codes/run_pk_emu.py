#!/bin/bash/python

import numpy as np
import subprocess as sp

# create franken-emu outputs for a given cosmology / z range
def create_pk(pk_outfile,params,j,num_params,zmin,zmax,nbins):
    dzl = (zmax-zmin)/nbins
    zl_array = np.linspace(zmin,zmax-dzl,nbins)+dzl/2.0
    omegaM = params[j][0]
    omegaB = params[j][1]
    sigma8 = params[j][2]
    h = params[j][3]
    n_s = params[j][4]
    if num_params==7:
        z_m = params[j][5]
        fwhm = params[j][6]
    for i in xrange(nbins):
        filename = pk_outfile+'/'+str(i)+"_"+str(zl_array[i])+"_"+str(dzl)+"_cosmo_"+str(j)+"_pk.dat"
        cmd = "./wle_numba/emu.exe "+filename+" "+str(omegaB)+" "+str(omegaM)+" "+str(n_s)+" "+str(h*100)+" -1.0 "+str(sigma8)+" "+str(zl_array[i])
        sp.call(cmd,shell=True)
    return 0


