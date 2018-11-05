#http://hmf.readthedocs.io/en/latest/
#http://hmf.readthedocs.io/en/stable/api_docs/hmf.html

#http://hmf.readthedocs.io/en/latest/_exampledoc/deal_with_cosmology.html#passing-custom-parameters

import matplotlib.pyplot as plt
import numpy as np
import itertools

import SetPub
SetPub.set_pub()

from hmf import cosmo
from hmf import MassFunction

from astropy.cosmology import LambdaCDM

from cycler import cycler
plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'k', 'gray']) +
                           cycler('linestyle', ['-', '-', '-', '-', '-', '-'])))



xlim1 = 10
xlim2 = 15
#bins 
Mass = np.logspace(np.log10(10**xlim1), np.log10(10**xlim2), 500)

AllCombinations = np.loadtxt('../../Pk_data/CosmicEmu-master/P_cb/xstar_halofit.dat')

#AllOm = np.linspace(0.25, 0.35, 11)

#delta_h = np.linspace(1.4, 1.7, 4)

y = np.zeros_like(Mass[::10])
y = np.hstack([1, 1, 1, 1, 1, y])

#AllOm = [0.2678]
#delta_h = [1.686]



#for Om, dc in itertools.izip( AllOm, delta_h):

for idx in range(np.shape(AllCombinations)[0]):
    Om_a, H0_a, ns_a, sigma8_a, delta_a = AllCombinations[idx]

#for Om, dc in [(xi,yj) for xi in AllOm for yj in delta_h]:
    #print x, y
    print(Om_a, H0_a, ns_a, sigma8_a, delta_a)
    print '-----------------'
# Standard Cosmology
    #HaloMF = MassFunction(cosmo_model = cosmo.WMAP5)   
    #my_cosmo = cosmo.Cosmology(cosmo_model=cosmo.WMAP5)

# Custom cosmology
    #new_model = LambdaCDM(H0 = 75.0, Om0= 0.4, Tcmb0 = 2.7, Ob0 = 0.3, Ode0=0.4)
    new_model = LambdaCDM(H0 = H0_a, Om0= Om_a, Tcmb0 = 2.7, Ob0 = 0.1, Ode0=1-Om_a)
    HaloMF = MassFunction(cosmo_model = new_model, delta_h = delta_a, sigma_8 = sigma8_a)
    HaloMF.update(n = ns_a)  
    my_cosmo = cosmo.Cosmology(cosmo_model = new_model)



				k = np.logspace(np.log(1e-3), np.log(5), 100)
    delta_k = 
    sigma_8 = 0.8
    z = 0.0
				hmf.halofit.halofit(k, delta_k, sigma_8, z, cosmo=None, takahashi=True)

    print HaloMF.parameter_values    # Check for parameters properly

    cumulative_mass_func = HaloMF.ngtm
    xxy =  np.hstack( [Om_a, H0_a, ns_a, sigma8_a, delta_a , cumulative_mass_func[::10]] )

    y = np.vstack( (y, xxy) )

    plt.figure(1)
    #plt.plot(Mass, mass_func)
    plt.plot(Mass, cumulative_mass_func, alpha = 0.1, lw = 2)
    
    
y = y[1:,]

#y = np.vstack(  (AllOm, a.T)  )

#ThetaI = np.empty([2,1])

#for Om, dc in [(xi,yj) for xi in AllOm for yj in delta_h]:
#    #print x, y
#    print(Om, dc)
#    ThetaI = np.vstack( ( ThetaI , np.array([Om, dc]).T ) )
#    print '-----------------'




#np.savetxt('HMF_5Para.txt', y)
    
    
#print HaloMF.parameter_info()

print 
#my_cosmo = cosmo.Cosmology()
print "Matter density: ", my_cosmo.cosmo.Om0
print "Hubble constant: ", my_cosmo.cosmo.H0
print "Dark Energy density: ", my_cosmo.cosmo.Ode0
print "Baryon density: ",  my_cosmo.cosmo.Ob0
print "Curvature density: ", my_cosmo.cosmo.Ok0

plt.figure(1)
plt.xscale('log')
plt.yscale('log')
#plt.xlim(1e12,6e14)
#plt.ylim(9e-7, 5e-3)
plt.ylabel(r"$n(M)(h^3/Mpc^3)$")
plt.xlabel("M $(h^{-1} M_\odot)$")
plt.legend(loc = "lower left")
plt.show()
#plt.savefig('plots/hmf_OmegaM', bbox_inches='tight')



# Check if log scale is working properly
plt.figure(10)
plt.plot(Mass)
plt.plot(Mass[::10], 'o')
plt.yscale('log')
plt.show()


from hmf.transfer import Transfer


teh_nl_tk = Transfer(transfer_model="EH", takahashi=True, z= 8.0)
teh_nl_ntk = Transfer(transfer_model="EH", takahashi=False, z=8.0)

teh_nl_tk.update(z=0)


plt.plot(teh_nl_tk.k, np.abs(teh_nl_ntk.nonlinear_power/teh_nl_tk.nonlinear_power -1))
#plt.plot(teh_nl_ntk.k, teh_nl_ntk.nonlinear_power)
#plt.plot(teh_nl_ntk.k, teh_nl_ntk.power)

plt.xscale('log')
#plt.yscale('log')
plt.grid(True)
