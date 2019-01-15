"""
Requires the following installations:

1. R (R studio is the easiest option: https://www.rstudio.com/products/rstudio/download/).
Installing R packages is easy, in R studio, command install.packages("package_name") works
(https://www.dummies.com/programming/r/how-to-install-load-and-unload-packages-in-r/)
The following R packages are required:
    1a. RcppCNPy
    1b. DiceKriging
    1c. GPareto

2. rpy2 -- which runs R under the hood (pip install rpy2 should work)
# http://rpy.sourceforge.net/rpy2/doc-2.1/html/index.html
# conda install -c r rpy2


Real Space

"""

##### Generic packages ###############
import numpy as np
import matplotlib.pylab as plt
import time
import glob
import os
import SetPub
SetPub.set_pub()

###### R kernel imports from rpy2 #####
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import r
from rpy2.robjects.packages import importr

############################# PARAMETERS ##############################

dirIn = "/home/nes/Desktop/AstroVAE/WL_emu/Codes/deprecated_codes/cl_outputs/"  ## Input Cl files
paramIn = "/home/nes/Desktop/AstroVAE/WL_emu/Codes/lhc_128.txt"  ## 8 parameter file
nRankMax = 4 ## Number of basis vectors in truncated PCA
GPmodel = '"RModels/R_GP_model_flat24' + str(nRankMax) + '.RData"'  ## Double and single quotes
# are necessary
# num_holdout = 4
################################# I/O #################################
RcppCNPy = importr('RcppCNPy')
# RcppCNPy.chooseCRANmirror(ind=1) # select the first mirror in the list


### 8 parameters:

# filelist = os.listdir(dirIn)
# filelist = glob.glob(dirIn + 'cls*')
filelist = glob.glob(dirIn + 'flat*')
filelist = sorted(filelist, key=lambda x: int(os.path.splitext(x)[0][74:]))

Px_flat = np.array([np.loadtxt(f) for f in filelist])


xvals = np.loadtxt(dirIn + 'xvals.txt')
### Px_flatnan = np.unique(np.array(np.argwhere(np.isnan(Px_flat)) )[:,0])

Px_flat = Px_flat[: ,:, 1]


nan_idx = [~np.isnan(Px_flat).any(axis=1)]
Px_flat = Px_flat[nan_idx]
# Px_flat = np.log(Px_flat[:, ::2])
Px_flat = np.log10(Px_flat[:, ::])


# np.random.seed(12)
# rand_idx = np.random.randint(0, np.shape(Px_flat)[0], np.shape(Px_flat)[0])

# Px_flat = Px_flat[rand_idx, :]

del_idx =  [15, 25, 80, 110]
Px_flat = np.delete(Px_flat, del_idx, axis = 0)




nr, nc = Px_flat[:,:].shape
y_train = ro.r.matrix(Px_flat[:,:], nrow=nr, ncol=nc)

# nr, nc = Px_flat[num_holdout:, :].shape
# y_train = ro.r.matrix(Px_flat[num_holdout:, :], nrow=nr, ncol=nc)



ro.r.assign("y_train2", y_train)
r('dim(y_train2)')

parameter_array = np.loadtxt(paramIn)
parameter_array = parameter_array[nan_idx]


parameter_array = np.delete(parameter_array, del_idx, axis = 0)


nr, nc = parameter_array[:,:].shape
u_train = ro.r.matrix(parameter_array[:,:], nrow=nr, ncol=nc)

# parameter_array = parameter_array[rand_idx, :]

# nr, nc = parameter_array[num_holdout:, :].shape
# u_train = ro.r.matrix(parameter_array[num_holdout:, :], nrow=nr, ncol=nc)

ro.r.assign("u_train2", u_train)
r('dim(u_train2)')


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


PCA_decomp()
GP_fit()


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

    return y_recon[0]


##################################### TESTING ##################################


PlotPrior = True

if PlotPrior:

    plt.rcParams['axes.color_cycle'] = [ 'navy', 'forestgreen', 'darkred', 'gold']


    plt.rc('text', usetex=True)  # Slower
    plt.rc('font', size=12)  # 18 usually

    plt.figure(999, figsize=(7, 6))
    from matplotlib import gridspec

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    gs.update(hspace=0.02, left=0.2, bottom=0.15)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.set_ylabel(r'$P(x)$',  fontsize = 15)

    # ax1.axhline(y=-1e-6, ls='dashed')
    # ax1.axhline(y=1e-6, ls='dashed')

    ax1.set_xlabel(r'$x$(arcmin)',  fontsize = 15)
    ax1.axhline(y=0, ls='dashed')


    ax0.set_yscale('log', base = 10)
    ax0.set_xscale('log', base = 10)
    ax1.set_xscale('log', base = 10)

    ax1.set_ylabel(r'emu/real - 1')
    # ax1.set_ylim(-5e-3, 5e-3)

    ax0.plot(xvals, 10**Px_flat.T, alpha=0.03, color='k')


# ax1.set_ylim(-5e-4, 5e-4)


ax0.set_xlim(xvals[0], xvals[-1])
ax1.set_xlim(xvals[0], xvals[-1])

ax0.set_xticklabels([])


color_id = 0
for x_id in del_idx:
    color_id = color_id + 1
#


# for x_id in [13, 24, 64, 83, 109]:
# for x_id in range(0, num_holdout):

    time0 = time.time()
    x_decodedGPy = GP_predict(parameter_array[x_id])  ## input parameters
    time1 = time.time()
    print('Time per emulation %0.2f' % (time1 - time0), ' s')
    x_test = Px_flat[x_id]

    ax0.plot(xvals, 10**x_decodedGPy, alpha=1.0, ls='--', label='emu', color=plt.cm.Set1(color_id))
    ax0.plot(xvals, 10**x_test, alpha=0.9, label='real', color=plt.cm.Set1(color_id))
    plt.legend()

    ax1.plot( xvals, (10**x_decodedGPy[:]) / (10**x_test[:])  - 1, color=plt.cm.Set1(color_id))

plt.show()


######### TEMPLATE FOR MCMC LIKELIHOOD FUNCTION #######################
# For emcee

def lnlike(theta, x, y, yerr):
    p1, p2, p3, p4, p5 = theta

    new_params = np.array([p1, p2, p3, p4, p5])

    model = GP_predict(new_params)
    # return -0.5 * (np.sum(((y - model) / yerr) ** 2.))
    return -0.5 * (np.sum(((y - model) / yerr) ** 2.))


