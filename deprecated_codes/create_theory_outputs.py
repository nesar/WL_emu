#!/bin/bash/python

import numpy as np
import subprocess as sp

# create latin hypercube parameters 
# syntax is a bit clugey, but can be fixed later
import os 
os.system('python ../latinHyp.py')

# Loading latin hypercube parameter files
params = np.loadtxt('lhc_128.txt')
#print(params)
#print(params[0])
print(len(params))

# create franken-emu outputs for these cosmologies/ z ranges
zmin = 0.0
zmax = 2.0
nbins = 1
dzl = (zmax-zmin)/nbins
zl_array = np.linspace(zmin,zmax-dzl,nbins)+dzl/2.0


output_dir = 'pk_outputs/'
for j in range(len(params)):
    omegaM = params[j][0]
    omegaB = params[j][1]
    sigma8 = params[j][2]
    print(sigma8)
    h = params[j][3]
    n_s = params[j][4]
    z_m = params[j][5]
    fwhm = params[j][6]
    for i in xrange(nbins):
        filename = output_dir+str(i)+"_"+str(zl_array[i])+"_"+str(dzl)+"_cosmo_"+str(j)+"_pk.dat"
        cmd = "./wle_numba/emu.exe "+filename+" "+str(omegaB)+" "+str(omegaM)+" "+str(n_s)+" "+str(h*100)+" -1.0 "+str(sigma8)+" "+str(zl_array[i])
        sp.call(cmd,shell=True)

# note - re-download frankenEmu, this doesn't seem like the most up to date parameter definitions for it


