##### Generic packages ###############
import numpy as np
import matplotlib.pylab as plt
import time
import glob
import os

##### MCMC ###########################

import emcee
import pygtc

###### R kernel imports from rpy2 #####
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import r
from rpy2.robjects.packages import importr


verbose=False

############################# PARAMETERS ##############################

#dirIn = "/home/nes/Desktop/AstroVAE/WL_emu/Codes/deprecated_codes/cl_outputs/"  ## Input Cl files
#paramIn = "/home/nes/Desktop/AstroVAE/WL_emu/Codes/lhc_128.txt"  ## 8 parameter file
nRankMax = 32  ## Number of basis vectors in truncated PCA
GPmodel = '"R_GP_model_flat' + str(nRankMax) + '.RData"'  ## Double and single quotes are necessary



########################### PCA ###################################
def PCA_decomp():
    Dicekriging = importr('DiceKriging')
    r('require(foreach)')
    # r('nrankmax <- 32')   ## Number of components
    ro.r.assign("nrankmax", nRankMax)

    r('svd(y_train2)')
    r('svd_decomp2 <- svd(y_train2)')
    r('svd_weights2 <- svd_decomp2$u[, 1:nrankmax] %*% diag(svd_decomp2$d[1:nrankmax])')


######################## GP FITTING ################################

## Build GP models
# This is evaluated only once for the file name. GP fitting is not required if the file exists.

def GP_fit():
    GPareto = importr('GPareto')

    ro.r('''

    GPmodel <- gsub("to", "",''' + GPmodel + ''')

    ''')

    r('''if(file.exists(GPmodel)){
            load(GPmodel)
        }else{
            models_svd2 <- list()
            for (i in 1: nrankmax){
                mod_s <- km(~., design = u_train2, response = svd_weights2[, i])
                models_svd2 <- c(models_svd2, list(mod_s))
            }
            save(models_svd2, file = GPmodel)

         }''')

    r('''''')




######################## GP PREDICTION ###############################


def GP_predict(para_array):
    ### Input: para_array -- 1D array [rho, sigma, tau, sspt]
    ### Output P(x) (size= 100)

    para_array = np.expand_dims(para_array, axis=0)

    nr, nc = para_array.shape
    Br = ro.r.matrix(para_array, nrow=nr, ncol=nc)

    ro.r.assign("Br", Br)

    r('wtestsvd2 <- predict_kms(models_svd2, newdata = Br , type = "UK")')
    r('reconst_s2 <- t(wtestsvd2$mean) %*% t(svd_decomp2$v[,1:nrankmax])')

    y_recon = np.array(r('reconst_s2'))

    y_recon = 10**(y_recon)  ## GP fitting was done in log space, exponentiating here

    return y_recon[0]


################ MAIN FUNC #####################


def create_emu(dirIn,paramIn):


    l = np.loadtxt(dirIn + 'ls_0.txt')
    RcppCNPy = importr('RcppCNPy')
    # RcppCNPy.chooseCRANmirror(ind=1) # select the first mirror in the list

    ### 8 parameters:

    # filelist = os.listdir(dirIn)
    filelist = glob.glob(dirIn + 'cls*')
    #filelist = glob.glob(dirIn + 'flat*')
    filelist = sorted(filelist, key=lambda x: int(os.path.splitext(x)[0][74:]))

    Px_flatflat = np.array([np.loadtxt(f) for f in filelist])
    ### Px_flatnan = np.unique(np.array(np.argwhere(np.isnan(Px_flatflat)) )[:,0])
    Px_flatflat = Px_flatflat[: ,:, 1]


    nan_idx = [~np.isnan(Px_flatflat).any(axis=1)]
    Px_flatflat = Px_flatflat[nan_idx]
    Px_flatflat = np.log10(Px_flatflat)

    nr, nc = Px_flatflat.shape
    y_train = ro.r.matrix(Px_flatflat, nrow=nr, ncol=nc)

    ro.r.assign("y_train2", y_train)
    r('dim(y_train2)')

    parameter_array = np.loadtxt(paramIn)
    parameter_array = parameter_array[nan_idx]

    nr, nc = parameter_array.shape
    u_train = ro.r.matrix(parameter_array, nrow=nr, ncol=nc)

    ro.r.assign("u_train2", u_train)
    r('dim(u_train2)')

    PCA_decomp()
    GP_fit()




##################################### TESTING ##################################

if verbose:
    plt.rc('text', usetex=True)  # Slower
    plt.rc('font', size=12)  # 18 usually
    
    plt.figure(999, figsize=(7, 6))
    from matplotlib import gridspec

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    gs.update(hspace=0.02, left=0.2, bottom=0.15)


    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.set_ylabel(r'$P(x)$ (flat)')

    ax1.axhline(y=1, ls='dotted')

    ax1.set_xlabel(r'$x$')

    ax0.set_xscale('log')
    ax1.set_xscale('log')
    ax0.set_yscale('log')
    ax1.set_ylabel(r'emu/real - 1')
    ax1.set_ylim(-1e-5, 1e-5)

    ax0.plot(l, 10**(Px_flatflat.T), alpha=0.03, color='k')

    for x_id in [3, 23, 43, 64, 83, 109]:
        time0 = time.time()
        x_decodedGPy = GP_predict(parameter_array[x_id])  ## input parameters
        time1 = time.time()
        print('Time per emulation %0.2f' % (time1 - time0), ' s')
        x_test = Px_flatflat[x_id]
        ax0.plot(l, x_decodedGPy, alpha=1.0, ls='--', label='emu')
        ax0.plot(l, 10**(x_test), alpha=0.9, label='real')
        plt.legend()
        ax1.plot(x_decodedGPy[1:] / 10**x_test[1:] - 1)





