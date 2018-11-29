#!/bin/bash/python

import numpy as np
import subprocess as sp
import altered_file
import matplotlib.pyplot as plt
import camb
import camb.correlations
# create latin hypercube parameters 
# syntax is a bit clugey, but can be fixed later
import os 
os.system('python ../latinHyp.py');
print("computed latin hypercube")

# Loading latin hypercube parameter files
params = np.loadtxt('lhc_128.txt')


# create franken-emu outputs for these cosmologies/ z ranges
zmin = 0.0
zmax = 2.0
nbins = 100
dzl = (zmax-zmin)/nbins
zl_array = np.linspace(zmin,zmax-dzl,nbins)+dzl/2.0

min_sep2 = 1.0
max_sep2 = 100
nbins2= 100

os.system('rm pk_outputs/*');
os.system('rm cl_outputs/*');
output_dir = 'pk_outputs/'
output_dir_cl = 'cl_outputs/'
for j in range(len(params)):
    print(j)
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
    l,c = altered_file.multiple_zs(omegaM,h*100,z_m,fwhm)
    os.system('rm pk_outputs/*');
    l = l.astype(int)
    np.savetxt(output_dir_cl+"cls_"+str(j)+".txt",c,fmt='%.5e')
    pp3_2 = np.zeros((10000,4))
    pp3_2[:, 1] = c[:] * (l* (l + 1.)) / (2. * np.pi)
    xvals = np.logspace(np.log10(min_sep2), np.log10(max_sep2), nbins2)
    cxvals = np.cos(xvals / (60.) / (180. / np.pi))
    vals = camb.correlations.cl2corr(pp3_2, cxvals)
    #np.savetxt(xvals,fmt='%.5e')
    np.savetxt(output_dir_cl+'flatP_'+str(j)+'.txt',vals,fmt='%.5e')
np.savetxt(output_dir_cl+'xvals.txt',xvals,fmt = '%d')
np.savetxt(output_dir_cl+"ls.txt",l,fmt='%d')

print("finished creating power spectra")



# note - re-download frankenEmu, this doesn't seem like the most up to date parameter definitions for it


