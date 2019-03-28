import matplotlib.pyplot as plt
import numpy as np
import itertools

import SetPub
SetPub.set_pub()

import hmf
from hmf import cosmo
from hmf import MassFunction

from astropy.cosmology import LambdaCDM




AllCombinations = np.loadtxt('../../../Pk_data/CosmicEmu-master/P_cb/xstar_32.dat')


for idx in range(np.shape(AllCombinations)[0]):
    Om_a, H0_a, ns_a, sigma8_a, delta_a = AllCombinations[idx][:5]

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

    hmf.halofit.halofit(k, delta_k, sigma_8, z, cosmo=my_cosmo, takahashi=True)
    hmf.halofit.halofit(k, delta_k, sigma_8, z, cosmo=None, takahashi=True)
    print HaloMF.parameter_values    # Check for parameters properly

    cumulative_mass_func = HaloMF.ngtm
    xxy =  np.hstack( [Om_a, H0_a, ns_a, sigma8_a, delta_a , cumulative_mass_func[::10]] )

    y = np.vstack( (y, xxy) )

    plt.figure(1)
#plt.plot(Mass, mass_func)
    plt.plot(Mass, cumulative_mass_func, alpha = 0.1, lw = 2)




