#!/bin/bash/python

#
#  This code comprises a wrapper function to create the theory outputs for some arbitrary n(z) distribution, build  
#  an emulator off these theory outputs, and run some basic applications of such an emulator. 
#
#  The emulator positions are chosen to be a linearly sampled latin hypercube of the input parameters. New parameters, 
#  for example bias parameters, can be added in in a simple manner - we have shown an example of this by inputting a 
#  gaussian n(z) distribution. 
#
#  This is currently a simple example of what such emulators can do, it can be expanded on using other simulation 
#  statistics. 
#
#
#  Built by P. Larsen and N. Ramachandra based on work by N. Mudur and the cosmicEmu team. 
#

import os
import numpy as np
import run_pk_emu
import altered_file

################

# create latin hypercube parameters - here we have 64 evaluations over 5 parameters
# 5 cosmology parameters so far, extra 2 possible for redshift distribution

os.system('python ../latinHyp.py 32 5');
print("computed latin hypercube")

# Loading latin hypercube parameter files
params = np.loadtxt('lhc_32_5.txt')


################
# imported run_pk_emu to run the franken-emu emulator
# then altered_file computes the theory codes at these positions

output_dir_cl = 'cl_outputs_g'

# Run theory code at latin hypercube positions - note this is currently only the full-sky version
for j in range(len(params)):
    os.system('rm pk_outputs_g/*');
    run_pk_emu.create_pk('pk_outputs_g',params,0,5,0.0,2.0,100)
    l,c = altered_file.multiple_zs(params[j][0],params[j][3]*100,input_nz=True)
    l = l.astype(int)
    np.savetxt(output_dir_cl+"/cls_"+str(j)+".txt",c[1:],fmt='%.5e')
    np.savetxt(output_dir_cl+"/ls_"+str(j)+".txt",l[1:],fmt='%d')

print("finished computing theory power spectra")

################

# Now we create the emulator from these theory outputs

import gp_emulation

gp_emulation.create_emu(output_dir_cl+'/','lhc_32_5.txt')

