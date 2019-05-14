"""
Python implementation of GP emulation.
"""

##### Generic packages ###############
import numpy as np
import matplotlib.pylab as plt
import time
import glob
import os

import SetPub

SetPub.set_pub()

import pickle

############################# PARAMETERS ##############################

dirIn = "/Users/nramachandra/Desktop/Projects/WL_emu/Codes/deprecated_codes/cl_outputs/"  ## Input Cl files
paramIn = "./Codes/lhc_128.txt"  ## 8 parameter file
nRankMax = 32  ## Number of basis vectors in truncated PCA

num_holdout = 4
################################# I/O #################################


filelist = glob.glob(dirIn + 'cls*')
filelist = sorted(filelist, key=lambda x: int(os.path.splitext(x)[0][82:]))

Cls = np.array([np.loadtxt(f) for f in filelist])

# Cl_nan = np.unique(np.array(np.argwhere(np.isnan(Cls)) )[:,0])
nan_idx = [~np.isnan(Cls).any(axis=1)]

Cls = Cls[nan_idx]

l = np.arange(Cls.shape[1])[1:]

Cls = np.log10(Cls[:, 1::])


del_idx = [20, 55, 60, 75]
Cls = np.delete(Cls, del_idx, axis=0)


nr, nc = Cls.shape

parameter_array = np.loadtxt(paramIn)
parameter_array = parameter_array[nan_idx]

parameter_array = np.delete(parameter_array, del_idx, axis=0)

########################### PCA ###################################
# set up pca compression
from sklearn.decomposition import PCA



######################## GP FITTING ################################

## Build GP models
# This is evaluated only once for the file name. GP fitting is not required if the file exists.
import GPy

######################## GP PREDICTION ###############################


def GPy_predict(para_array, GPmodel='GPy_model'):
    m1 = GPy.models.GPRegression.load_model(GPmodel + '.zip')

    m1p = m1.predict(para_array)  # [0] is the mean and [1] the predictive
    W_predArray = m1p[0]
    W_varArray = m1p[1]
    return W_predArray, W_varArray


def Emu(para_array, PCAmodel):
    pca_model = pickle.load(open(PCAmodel, 'rb'))

    W_predArray, _ = GPy_predict(para_array, 'GPy_model')
    x_decoded = pca_model.inverse_transform(W_predArray)

    return x_decoded[0]



######################## GP PREDICTION ###############################


def GPy_predict(para_array, GPmodel='GPy_model'):
    m1 = GPy.models.GPRegression.load_model(GPmodel + '.zip')

    m1p = m1.predict(para_array)  # [0] is the mean and [1] the predictive
    W_predArray = m1p[0]
    W_varArray = m1p[1]
    return W_predArray, W_varArray


def Emu(para_array, PCAmodel, GPmodel):
    pca_model = pickle.load(open(PCAmodel, 'rb'))

    if len(para_array.shape) == 1:

        W_predArray, _ = GPy_predict(np.expand_dims(para_array, axis=0), GPmodel)
        x_decoded = pca_model.inverse_transform(W_predArray)

        return x_decoded[0]

    else:

        W_predArray, _ = GPy_predict(para_array, GPmodel)
        x_decoded = pca_model.inverse_transform(W_predArray)

        return x_decoded.T



################################################################################


x_decoded2 = Emu(np.expand_dims(parameter_array[del_idx][0], axis=0), PCAmodel='PCA_model', GPmodel='GPy_model')




plt.figure(132)

plt.loglog(l, 10 ** x_decoded2, '--')

##################################### TESTING ##################################

### PRIOR PLOT ####

PlotPrior = False

if PlotPrior:
    plt.rc('text', usetex=True)  # Slower
    plt.rc('font', size=12)  # 18 usually

    plt.figure(999, figsize=(8, 6))
    from matplotlib import gridspec

    gs = gridspec.GridSpec(1, 1)
    gs.update(hspace=0.02, left=0.2, bottom=0.15)
    ax0 = plt.subplot(gs[0])
    # ax1 = plt.subplot(gs[1])

    ax0.set_ylabel(r'$C_l$', fontsize=15)

    # ax1.axhline(y=0, ls='dashed')
    # ax1.axhline(y=-1e-6, ls='dashed')
    # ax1.axhline(y=1e-6, ls='dashed')

    ax0.set_xlabel(r'$l$', fontsize=15)

    ax0.set_xscale('log', base=10)
    # ax1.set_xscale('log')

    # ax1.set_ylabel(r'emu/real - 1')
    # ax1.set_ylim(-2e-4, 2e-4)

    ax0.plot(l, Cls.T, alpha=0.3)

    plt.savefig('Plots/ClEmuPrior.png', figsize=(24, 18), bbox_inches="tight", dpi=900)
    plt.show()

#### POSTERIOR DRAWS ######

# plt.rcParams['axes.color_cycle'] = [ 'navy', 'forestgreen', 'darkred', 'sienna']


plt.rc('text', usetex=True)  # Slower
plt.rc('font', size=12)  # 18 usually

plt.figure(999, figsize=(7, 6))
from matplotlib import gridspec

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(hspace=0.02, left=0.2, bottom=0.15)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_ylabel(r'$C_l$', fontsize=15)

ax1.axhline(y=0, ls='dashed')
# ax1.axhline(y=-1e-6, ls='dashed')
# ax1.axhline(y=1e-6, ls='dashed')

ax1.set_xlabel(r'$l$', fontsize=15)

ax0.set_yscale('log', base=10)
ax0.set_xscale('log', base=10)
ax1.set_xscale('log', base=10)

ax1.set_ylabel(r'emu/real - 1')
ax1.set_ylim(-5e-4, 5e-4)

ax0.plot(l, 10 ** Cls.T, alpha=0.03, color='k')

color_id = 0
for x_id in del_idx:
    color_id = color_id + 1
    #
    # for x_id in range(0, num_holdout):

    # time0 = time.time()
    # x_decodedGPy = GP_predict(parameter_array[x_id])  ## input parameters
    # time1 = time.time()
    # ax0.plot(l, 10 ** x_decodedGPy, alpha=1.0, ls='--', label='emu', color=plt.cm.Set1(color_id))

    time0 = time.time()
    x_decoded_new = Emu(parameter_array[x_id], PCAmodel='PCA_model', GPmodel='GPy_model')

    time1 = time.time()
    print('Time per emulation %0.2f' % (time1 - time0), ' s')

    ax0.plot(l, 10 ** x_decoded_new, alpha=1.0, ls='--', label='emu', color=plt.cm.Set1(color_id))

    x_test = Cls[x_id]
    ax0.plot(l, 10 ** x_test, alpha=0.9, label='real', color=plt.cm.Set1(color_id))

    ax1.plot(l, (10 ** x_decoded_new) / (10 ** x_test) - 1, ls='--', color=plt.cm.Set1(color_id))

    plt.legend()




    # ax1.plot(  (x_decodedGPy[1:])/(x_test[1:]) - 1)

ax0.set_xlim(l[0], l[-1])
ax1.set_xlim(l[0], l[-1])

ax0.set_xticklabels([])

# plt.savefig('Plots/ClEmuGPy.png', figsize=(28, 24), bbox_inches="tight", dpi=900)
plt.savefig('Plots/ClEmuGPy.png', figsize=(28, 24), bbox_inches="tight")

plt.show()

#### Plot PCA bases and weights ####



######### TEMPLATE FOR MCMC LIKELIHOOD FUNCTION #######################
# For emcee

def lnlike(theta, x, y, yerr):
    p1, p2, p3, p4, p5 = theta

    new_params = np.array([p1, p2, p3, p4, p5])

    model = GPy_predict(new_params)
    # return -0.5 * (np.sum(((y - model) / yerr) ** 2.))
    return -0.5 * (np.sum(((y - model) / yerr) ** 2.))


