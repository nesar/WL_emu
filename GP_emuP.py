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


rgl issue in Tricia's Mac

"""


##### Generic packages ###############
import numpy as np
import matplotlib.pylab as plt
import time
import glob
import os

###### R kernel imports from rpy2 #####
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import r
from rpy2.robjects.packages import importr

############################# PARAMETERS ##############################

dirIn = "/home/nes/Desktop/AstroVAE/WL_emu/Codes/deprecated_codes/cl_outputs/"   ## Input Cl files
paramIn = "/home/nes/Desktop/AstroVAE/WL_emu/Codes/lhc_128.txt"   ## 8 parameter file
nRankMax = 16    ## Number of basis vectors in truncated PCA
GPmodel = '"R_GP_model1' + str(nRankMax) + '.RData"'  ## Double and single quotes are necessary

################################# I/O #################################
RcppCNPy = importr('RcppCNPy')
# RcppCNPy.chooseCRANmirror(ind=1) # select the first mirror in the list


### 8 parameters: 

# filelist = os.listdir(dirIn)
# ### dirIn contains ONLY the cls_*.txt files
# filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0][4:]))
#
#
# Cls = np.array([np.loadtxt(dirIn + f) for f in filelist])


filelist = glob.glob(dirIn + 'cls*')
filelist = sorted(filelist, key=lambda x: int(os.path.splitext(x)[0][72:]))

Cls = np.array([np.loadtxt(f) for f in filelist])



# Cl_nan = np.unique(np.array(np.argwhere(np.isnan(Cls)) )[:,0])
nan_idx = [~np.isnan(Cls).any(axis=1)]

Cls = Cls[nan_idx]

Cls = np.log(Cls[:, 1::10])


nr, nc = Cls[4:, :].shape
y_train = ro.r.matrix(Cls[4:, :], nrow=nr, ncol=nc)

ro.r.assign("y_train2", y_train)
r('dim(y_train2)')


parameter_array = np.loadtxt(paramIn)
parameter_array = parameter_array[nan_idx]

nr, nc = parameter_array[4:, :].shape
u_train = ro.r.matrix(parameter_array[4:, :], nrow=nr, ncol=nc)

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


plt.rc('text', usetex=True)   # Slower
plt.rc('font', size=12)  # 18 usually

plt.figure(999, figsize=(7, 6))
from matplotlib import gridspec

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(hspace=0.02, left=0.2, bottom=0.15)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_ylabel(r'$C_l$')

ax1.axhline(y=1, ls='dotted')
# ax1.axhline(y=-1e-6, ls='dashed')
# ax1.axhline(y=1e-6, ls='dashed')

ax1.set_xlabel(r'$l$')

ax0.set_xscale('log')
ax1.set_xscale('log')

ax1.set_ylabel(r'emu/real - 1')
ax1.set_ylim(-2e-5, 2e-5)


ax0.plot(Cls.T, alpha = 0.03, color = 'k')

# for x_id in [3, 23, 43, 64, 83, 109]:

for x_id in range(0, 4):

    time0 = time.time()
    x_decodedGPy = GP_predict(parameter_array[x_id])  ## input parameters
    time1 = time.time()
    print('Time per emulation %0.2f'% (time1 - time0), ' s')
    x_test = Cls[x_id]

    ax0.plot(x_decodedGPy, alpha=1.0, ls='--', label='emu')
    ax0.plot(x_test, alpha=0.9, label='real')
    plt.legend()

    ax1.plot(x_decodedGPy[1:] / x_test[1:] - 1)

plt.show()



######### TEMPLATE FOR MCMC LIKELIHOOD FUNCTION #######################
# For emcee

def lnlike(theta, x, y, yerr):
    p1, p2, p3, p4, p5 = theta

    new_params = np.array([p1, p2, p3, p4, p5])

    model = GP_predict(new_params)
    # return -0.5 * (np.sum(((y - model) / yerr) ** 2.))
    return -0.5 * (np.sum(((y - model) / yerr) ** 2.))


