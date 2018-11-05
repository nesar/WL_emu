import numpy as np
import matplotlib.pylab as plt
import glob
import SetPub

SetPub.set_pub()


fileList = glob.glob('../../Pk_data/CosmicEmu-master/P_cb/EMU*.txt')

#plt.figure(1)

#for fileIn in fileList:
#    kPk = np.loadtxt(fileIn)
#    plt.plot(kPk[:,0], kPk[:,1], 'b', alpha = 0.3)


#plt.loglog()
#plt.xlabel(r'$k$ [h/Mpc]', fontsize=16)
#plt.ylabel(r'$P(k)$', fontsize=16)

#plt.show()



#lhd = np.loadtxt('../CosmicEmu-master/P_cb/xstar.dat')


#AllLabels = AllLabels = [r'$\tilde{\Omega}_m$', r'$\tilde{\Omega}_b$', r'$\tilde{h}$',
#                     r'$\tilde{n}_s$', 'a', 'b', 'c', 'd', 'e']


'''
print lhd.shape
plt.figure(32)

##
f, a = plt.subplots(8, 8, sharex=True, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.rcParams.update({'font.size': 8})

for i in range(8):
    for j in range(i+1):
        print(i,j)
        if(i!=j):
            a[i, j].scatter(lhd[:, i], lhd[:, j], s=20)
            a[i, j].grid(True)
        else:
            a[i, i].text(0.4, 0.4, AllLabels[i], size = 'xx-large')
            hist, bin_edges = np.histogram(lhd[:,i], density=True, bins=64)
            a[i,i].bar(bin_edges[:-1], hist/hist.max(), width=0.2)
            #plt.xlim(0,1)
            #plt.ylim(0,1)

plt.show()


'''





import numpy as np
import camb
import itertools
from camb import model, initialpower
import matplotlib.pylab as plt

import time
time0 = time.time()

"""
first 2 outputs from CAMB - totCL and unlensed CL both are 0's. 
CAMBFast maybe better?
CosmoMC works well with CAMB
http://camb.readthedocs.io/en/latest/CAMBdemo.html
https://wiki.cosmos.esa.int/planckpla2015/index.php/CMB_spectrum_%26_Likelihood_Code
"""

numpara = 5
# ndim = 2551
totalFiles =  8
# lmax0 = 2500


lmax0 = 2500   ## something off above 8250
# model.lmax_lensed.value = 8250 by default
ell_max = 1000

# ndim = lmax0 + 1

ndim = 351
z_range = [0.,]




#para5 = np.loadtxt('../Cl_data/Data/LatinCosmoP5'+str(totalFiles)+'.txt')

para5 = np.loadtxt('../../Pk_data/CosmicEmu-master/P_cb/xstar_halofit.dat')#[:totalFiles]

# print(para5)

'''

f, a = plt.subplots(5, 5, sharex=True, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.rcParams.update({'font.size': 8})


AllLabels = [r'$\tilde{\Omega}_m$', r'$\tilde{\Omega}_b$', r'$\tilde{\sigma}_8$', r'$\tilde{h}$',
             r'$\tilde{n}_s$'] ### n_eff, mass nutrino -- tau



for i in range(5):
    for j in range(i+1):
        print(i,j)
        # a[i,j].set_xlabel(AllLabels[i])
        # a[i,j].set_ylabel(AllLabels[j])
        if(i!=j):
            a[i, j].scatter(para5[:, i], para5[:, j], s=10)
            a[i, j].grid(True)
        else:
            # a[i,i].set_title(AllLabels[i])
            a[i, i].text(0.4, 0.4, AllLabels[i], size = 'xx-large')
            hist, bin_edges = np.histogram(para5[:,i], density=True, bins=64)
            # a[i,i].bar(hist)
            a[i,i].bar(bin_edges[:-1], hist/hist.max(), width=0.2)
            # plt.xlim(0,1)
            # plt.ylim(0,1)

            # n, bins, patches = a[i,i].hist(lhd[:,i], bins = 'auto', facecolor='b', alpha=0.25)
            # a[i, i].plot(lhd[:, i], 'go')

#plt.savefig('LatinSq.png', figsize=(10, 10))


plt.show()
'''



fig, subplt = plt.subplots(2, 1, sharex=True, figsize = (8,6))
from matplotlib import gridspec

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(hspace=0.05, left=0.2, bottom = 0.15)  # set the spacing between axes.
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])




#---------------------------------------
AllPkL = np.zeros(shape=(totalFiles, numpara + ndim) ) # linear
AllPkNL = np.zeros(shape=(totalFiles, numpara + ndim) ) # nonlinear


for i in range(totalFiles):
    print(i, para5[i])

    pars = camb.CAMBparams()

    # pars.set_cosmology(H0=100*para5[i, 2], ombh2=para5[i, 1], omch2=para5[i, 0], mnu=0.06, omk=0,
    #                    tau=0.06)

    pars.set_cosmology(H0=100*para5[i, 3], ombh2=para5[i, 1], omch2=para5[i, 0], mnu=0.06, omk=0,
                       tau=0.06)
				

    pars.set_dark_energy(w=-1.0, sound_speed=1.0, dark_energy_model='fluid')

    # pars.set_dark_energy()  # re-set defaults

    pars.InitPower.set_params(ns=para5[i, 4], r=0)
    # pars.set_for_lmax(lmax, lens_potential_accuracy=0);




    #-------- sigma_8 --------------------------
    pars.set_matter_power(redshifts=z_range, kmax=7.0)
    # Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-3, maxkh=5, npoints = ndim)
    s8 = np.array(results.get_sigma8())

    # Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-3, maxkh=5, npoints=ndim)
    #kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_interpolator(nonlinear=True, var1=None, var2=None, hubble_units=True, k_hunit=False, return_z_k=True, log_interp=True, extrap_kmax=None)

    sigma8_camb = results.get_sigma8()  # present value of sigma_8 --- check kman, mikh etc
    #---------------------------------------------------

    sigma8_input = para5[i, 2]

    r = (sigma8_input ** 2) / (sigma8_camb ** 2) # rescale factor
    # r = 1
    # #---------------------------------------------------

    for j, (redshift, line) in enumerate(zip(z, ['-', '--'])):
        ax0.loglog(kh, pk[j, :]*r, color='k', ls='-.', alpha=0.3)  # check multiplication by r
        ax0.loglog(kh_nonlin, pk_nonlin[j, :]*r, color='b', ls=line, alpha=0.3)

        kPk = np.loadtxt(fileList[i])
        ax0.loglog(kPk[:,0]/para5[i, 3], kPk[:,1], 'r', alpha = 0.3)  # check k/h and P(k/h) issue /para5[i, 3]

        ax1.loglog(kPk[:,0]/para5[i, 3], kPk[:,1]/(pk_nonlin[j, :]*r), 'b', alpha = 0.3)




    ax0.set_xlabel(r'$k/h$ Mpc', fontsize=16);
    ax0.set_ylabel(r'$P(k/h)$', fontsize=16);
    ax0.legend(['linear', 'non-linear: Halofit', 'nonlinear: CosmicEmu'], loc='lower left');
 
    ax1.set_xlabel(r'$k$ Mpc/h', fontsize=16)
    ax1.set_ylabel(r'$P(k)^{emu}$/$P(k)^{halofit}$', fontsize=16)
    # plt.title('Matter power at z=%s and z= %s' % tuple(z));


    #AllPkL[i] = np.hstack([para5[i],  pk_nonlin[0]*r])
    #AllPkNL[i] = np.hstack([para5[i],  pk_nonlin[0]*r])


#np.savetxt('../Pk_data/Data/P5kh_'+str(totalFiles)+'.txt', kh)
#np.savetxt('../Pk_data/Data/P5PkL_'+str(totalFiles)+'.txt', AllPkL)
#np.savetxt('../Pk_data/Data/P5PkNL_'+str(totalFiles)+'.txt', AllPkNL)

#fig.tight_layout()


def hmf_halofit():
				#https://github.com/steven-murray/hmf/blob/master/development/halofit_testing.ipynb
				from hmf.transfer import Transfer


				teh_nl_tk = Transfer(transfer_model="EH", takahashi=True, z=0.0)
				teh_nl_ntk = Transfer(transfer_model="EH", takahashi=False, z=0.0)

				teh_nl_tk.update(z=0)


				#plt.plot(teh_nl_tk.k, np.abs(teh_nl_ntk.nonlinear_power/teh_nl_tk.nonlinear_power -1))
				ax0.loglog(teh_nl_ntk.k, teh_nl_ntk.nonlinear_power, 'k--')
				ax0.loglog(teh_nl_ntk.k, teh_nl_ntk.power, 'k-.')

				#plt.xscale('log')
				#plt.yscale('log')
				#plt.grid(True)




				

hmf_halofit()


plt.show()
