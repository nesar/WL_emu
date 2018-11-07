import numpy as np
import matplotlib.pylab as plt
import glob
import SetPub

SetPub.set_pub()

import numpy as np
import camb
import itertools
from camb import model, initialpower
import matplotlib.pylab as plt

import time
time0 = time.time()




numpara = 5

totalFiles =  2
# lmax0 = 2500


lmax0 = 2500   ## something off above 8250
# model.lmax_lensed.value = 8250 by default
ell_max = 1000

# ndim = lmax0 + 1

ndim = 351
z_range = [0, 0.3, 0.5, 0.9 ,1.0]

z_range = [0.0, ]

fileList = glob.glob('../../Pk_data/CosmicEmu-master/P_tot/EMU*.txt')

def plot_latin(fileIn):
    lhd = np.loadtxt(fileIn)


    AllLabels = AllLabels = [r'$\tilde{\Omega}_m$', r'$\tilde{\Omega}_b$', r'$\tilde{h}$',
                  r'$\tilde{n}_s$', 'a', 'b', 'c', 'd', 'e']

    print lhd.shape
    plt.figure(32)

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


para5 = np.loadtxt('../../Pk_data/CosmicEmu-master/P_cb/xstar_halofit.dat')#[:totalFiles]



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

    pars.set_cosmology(H0=100*para5[i, 3], ombh2=para5[i, 1], omch2=para5[i, 0] - para5[i, 1], mnu=0.06, omk=0,
                       tau=0.06)
				

    pars.set_dark_energy(w=-1.0, sound_speed=1.0, dark_energy_model='fluid')

    # pars.set_dark_energy()  # re-set defaults

    pars.InitPower.set_params(ns=para5[i, 4], r=0)
    # pars.set_for_lmax(lmax, lens_potential_accuracy=0);





    #-------- sigma_8 --------------------------
    pars.set_matter_power(redshifts=z_range, kmax=10.0)
    # Linear spectra
    pars.NonLinear = model.NonLinear_none
    
    
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-3/para5[i,3], maxkh=5/para5[i,3], npoints = ndim)
    s8 = np.array(results.get_sigma8())

    # Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-3/para5[i,3], maxkh=5/para5[i,3], npoints=ndim)
    #kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_interpolator(nonlinear=True, var1=None, var2=None, hubble_units=True, k_hunit=False, return_z_k=True, log_interp=True, extrap_kmax=None)

    sigma8_camb = results.get_sigma8()  # present value of sigma_8 --- check kman, mikh etc
    #---------------------------------------------------

    sigma8_input = para5[i, 2]

    r = (sigma8_input ** 2) / (sigma8_camb ** 2) # rescale factor
    r = 1
    # #---------------------------------------------------

    camb.set_halofit_version('takahashi')
    kh_nonlin_takahashi, _, pk_takahashi = results.get_nonlinear_matter_power_spectrum(params=pars)
    camb.set_halofit_version('mead')
    kh_nonlin_mead, _, pk_mead = results.get_nonlinear_matter_power_spectrum(params=pars)





    for j, (redshift, line) in enumerate(zip(z, ['-', '--'])):
        ax0.loglog(kh*para5[i,3], (pk[j, :]*r)/(para5[i,3]**3), color='k', ls='--', alpha=0.3)  # check multiplication by r
        ax0.loglog(kh_nonlin*para5[i,3], (pk_nonlin[j, :]*r)/(para5[i,3]**3), color='b', ls=line, alpha=0.3)
        ax0.loglog(kh_nonlin_takahashi*para5[i,3], (pk_takahashi[j, :]*r)/(para5[i,3]**3), color='orange', ls=line, alpha=0.3)
        ax0.loglog(kh_nonlin_mead*para5[i,3], (pk_mead[j, :]*r)/(para5[i,3]**3), color='green', ls=line, alpha=0.3)
        


        kPk = np.loadtxt(fileList[i])
        ax0.loglog(kPk[:,0], kPk[:,1], 'r-.', alpha = 0.9, lw = 2 )  # check k/h and P(k/h) issue /para5[i, 3]

        ax1.loglog(kPk[:,0], (kPk[:,1]/(pk_nonlin[j, :]*r))/(para5[i,3]**3), 'b', alpha = 0.3)


        



    ax0.set_xlabel(r'$k/h$ Mpc', fontsize=16);
    ax0.set_ylabel(r'$P(k/h)$', fontsize=16);
    ax0.legend(['linear', 'non-linear: Halofit', 'Takahasi', 'Mead', 'non-linear: CosmicEmu'], loc='lower left');

    ax1.set_xlabel(r'$k$ ', fontsize=16)
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



def takahasi():
    #The non-linear model can be changed like this:
    camb.set_halofit_version('takahashi')
    kh_nonlin, _, pk_takahashi = results.get_nonlinear_matter_power_spectrum(params=pars)
    camb.set_halofit_version('mead')
    kh_nonlin, _, pk_mead = results.get_nonlinear_matter_power_spectrum(params=pars)

    fig, axs=plt.subplots(2,1, sharex=True, figsize=(8,8))
    ax0.loglog(kh_nonlin*para5[i,3], pk_takahashi[0]/(para5[i,3]**3))
    ax0.loglog(kh_nonlin*para5[i,3], pk_mead[0]/(para5[i,3]**3))
    axs[1].semilogx(kh_nonlin, pk_mead[0]/pk_takahashi[0]-1)
    axs[1].set_xlabel(r'$k/h\, \rm{Mpc}$')    
    axs[1].legend(['Mead/Takahashi-1'], loc='upper left');




def get_interpolated():
    #For calculating large-scale structure and lensing results yourself, get a power spectrum
    #interpolation object. In this example we calculate the CMB lensing potential power
    #spectrum using the Limber approximation, using PK=camb.get_matter_power_interpolator() function.   
    #calling PK(z, k) will then get power spectrum at any k and redshift z in range.

    nz = 100 #number of steps to use for the radial/redshift integration
    kmax=10  #kmax to use
    #First set up parameters as usual
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965)

    #For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
    #so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts
    results= camb.get_background(pars)
    chistar = results.conformal_time(0)- model.tau_maxvis.value
    chis = np.linspace(0,chistar,nz)
    zs=results.redshift_at_comoving_radial_distance(chis)
    #Calculate array of delta_chi, and drop first and last points where things go singular
    dchis = (chis[2:]-chis[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]

    #Get the matter power spectrum interpolation object (based on RectBivariateSpline)
    #Here for lensing we want the power spectrum of the Weyl potential.

    #PK = camb.get_matter_power_interpolator(pars, nonlinear=True,hubble_units=False, k_hunit=False, kmax=kmax,var1=model.Transfer_Weyl,var2=model.Transfer_Weyl, zmax=zs[-1])

    PK = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=kmax, zmax = zs[-1]) 

    #Have a look at interpolated power spectrum results for a range of redshifts
    #Expect linear potentials to decay a bit when Lambda becomes important, and change from non-linear growth
    plt.figure(figsize=(8,5))
    k=np.exp(np.log(10)*np.linspace(-4,5,1000))
    #zplot = [0, 0.5, 1, 4 ,20]
    zplot = [0.0]
    #zplot = [0, 0.3, 0.5, 0.9 ,1.0]
    for z in zplot:
        ax0.loglog(k, PK.P(z,k))
        #plt.xlim([1e-4,kmax])
        #plt.xlabel('k Mpc')
        #plt.ylabel('$P_\Psi\, Mpc^{-3}$')
        plt.legend(['z=%s'%z for z in zplot]);
        #plt.show()





#hmf_halofit()
takahasi()
get_interpolated()


plt.show()
