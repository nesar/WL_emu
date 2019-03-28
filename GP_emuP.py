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


rgl issue in Tricia's Mac  -- resolved



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

dirIn = "/home/nes/Desktop/AstroVAE/WL_emu/Codes/deprecated_codes/cl_outputs/"   ## Input Cl files
paramIn = "/home/nes/Desktop/AstroVAE/WL_emu/Codes/lhc_128.txt"   ## 8 parameter file
nRankMax = 32    ## Number of basis vectors in truncated PCA
GPmodel = '"R_GP_model123a2' + str(nRankMax) + '.RData"'  ## Double and single quotes are necessary

num_holdout = 4
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

l = np.arange(Cls.shape[1])[1:]

# Cls = np.log(Cls[:, 1::2])
Cls = np.log10(Cls[:, 1::])


# np.random.seed(12)
# rand_idx = np.random.randint(0, np.shape(Cls)[0], np.shape(Cls)[0])

# Cls = Cls[rand_idx, :]
del_idx =  [20, 55, 60, 75]
Cls = np.delete(Cls, del_idx, axis = 0)


# nr, nc = Cls[num_holdout:, :].shape
# y_train = ro.r.matrix(Cls[num_holdout:, :], nrow=nr, ncol=nc)

nr, nc = Cls.shape
y_train = ro.r.matrix(Cls, nrow=nr, ncol=nc)



ro.r.assign("y_train2", y_train)
r('dim(y_train2)')


parameter_array = np.loadtxt(paramIn)
parameter_array = parameter_array[nan_idx]


# parameter_array = parameter_array[rand_idx, :]
parameter_array = np.delete(parameter_array, del_idx, axis = 0)


# nr, nc = parameter_array[num_holdout:, :].shape
# u_train = ro.r.matrix(parameter_array[num_holdout:, :], nrow=nr, ncol=nc)

nr, nc = parameter_array.shape
u_train = ro.r.matrix(parameter_array, nrow=nr, ncol=nc)

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


# set up pca compression
from sklearn.decomposition import PCA

def PCA_compress(x, nComp):
    # x is in shape (nCosmology, nbins)
    pca_model = PCA(n_components=nComp)
    principalComponents = pca_model.fit_transform(x)
    pca_bases = pca_model.components_

    print("original shape:   ", x.shape)
    print("transformed shape:", principalComponents.shape)
    print("bases shape:", pca_bases.shape)

    import pickle
    pickle.dump(pca_model, open('PCA_model', 'wb'))


    return pca_model, np.array(principalComponents), np.array(pca_bases)


# mean_peak_counts = np.mean(data_sim, axis=1)
# mean_peak_counts[0].shape

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

#
# def GP_fit_pca():
#     GPareto = importr('GPareto')
#
#     ro.r.assign("models_svd2_pca", pca_weights)
#     r('dim(models_svd2_pca)')
#
#     ro.r('''
#
#     GPmodel <- gsub("to", "",''' + GPmodel + ''')
#
#     ''')
#
#     r('''if(file.exists(GPmodel)){
#             load(GPmodel)
#         }else{
#             models_svd2_pca <- list()
#             for (i in 1: nrankmax){
#                 mod_s <- km(~., design = u_train2, response = svd_weights2[, i])
#                 models_svd2_pca <- c(models_svd2_pca, list(mod_s))
#             }
#             save(models_svd2_pca, file = GPmodel)
#
#          }''')
#
#     r('''''')



import GPy

def GPy_fit(parameter_array, weights, fname =  'GPy_model'):
    kern = GPy.kern.Matern52(5, 0.1)
    m1 = GPy.models.GPRegression(parameter_array, weights, kernel=kern)
    m1.Gaussian_noise.variance.constrain_fixed(1e-16)
    m1.optimize(messages=True)
    m1.save_model(fname, compress=True, save_data=True)



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


#
# def GP_predict_pca(para_array):
#     ### Input: para_array -- 1D array [rho, sigma, tau, sspt]
#     ### Output P(x) (size= 100)
#
#     para_array = np.expand_dims(para_array, axis=0)
#
#     nr, nc = para_array.shape
#     Br = ro.r.matrix(para_array, nrow=nr, ncol=nc)
#
#     ro.r.assign("Br", Br)
#
#     r('wtestsvd2 <- predict_kms(models_svd2_pca, newdata = Br , type = "UK")')
#     r('reconst_s2 <- t(wtestsvd2$mean) %*% t(svd_decomp2$v[,1:nrankmax])')
#
#     y_recon = np.array(r('reconst_s2'))
#
#     return y_recon[0]


def GPy_predict(para_array, GPmodel = 'GPy_model', PCAmodel = 'PCA_model'):

    m1 = GPy.models.GPRegression.load_model(GPmodel + '.zip')

    m1p = m1.predict(para_array)  # [0] is the mean and [1] the predictive
    W_predArray = m1p[0]
    W_varArray = m1p[1]

    # return W_predArray, W_varArray

    # if np.shape(para_array) == 1: np.expand_dims( parameter_array[del_idx][0], axis=0 )
    ### add this later



    import pickle

    pca_model = pickle.load(open(PCAmodel, 'rb'))

    # x_decoded = pca_model.inverse_transform(W_pred_mean[0])
    x_decoded = pca_model.inverse_transform(W_predArray)

    return x_decoded[0]


# def GPy_predict0(para_array, GPmodel='GPy_model'):
#     m1 = GPy.models.GPRegression.load_model(GPmodel + '.zip')
#
#     m1p = m1.predict(para_array)  # [0] is the mean and [1] the predictive
#     W_predArray = m1p[0]
#     W_varArray = m1p[1]
#
#     return W_predArray, W_varArray




################################################################################

PCA_decomp()
GP_fit()


# https://shankarmsy.github.io/posts/pca-sklearn.html
# pca_model, pca_weights, pca_bases = PCA_compress( np.array(y_train), nComp=nRankMax)



# svd_bases = r('svd_decomp2$v[,1:nrankmax]').T
# svd_weights  = r('svd_weights2')


# GP_fit_pca()
# GPy_fit(parameter_array, pca_weights)


# W_pred_mean, W_pred_var  = GPy_predict0( np.expand_dims( parameter_array[del_idx][0], axis=0 ))
#
# x_decoded = pca_model.inverse_transform(W_pred_mean[0])





# plt.plot(pca_model.explained_variance_ratio_)

x_decoded2 = GPy_predict(np.expand_dims( parameter_array[del_idx][0], axis=0 ))


plt.figure(132)

plt.loglog(l, 10**x_decoded2, '--')
# plt.loglog(l, 10**x_decoded)



##################################### TESTING ##################################

### PRIOR PLOT ####

PlotPrior = False

if PlotPrior:

    plt.rc('text', usetex=True)   # Slower
    plt.rc('font', size=12)  # 18 usually

    plt.figure(999, figsize=(8, 6))
    from matplotlib import gridspec

    gs = gridspec.GridSpec(1, 1)
    gs.update(hspace=0.02, left=0.2, bottom=0.15)
    ax0 = plt.subplot(gs[0])
    # ax1 = plt.subplot(gs[1])

    ax0.set_ylabel(r'$C_l$', fontsize = 15)

    # ax1.axhline(y=0, ls='dashed')
    # ax1.axhline(y=-1e-6, ls='dashed')
    # ax1.axhline(y=1e-6, ls='dashed')

    ax0.set_xlabel(r'$l$', fontsize = 15)

    ax0.set_xscale('log', base = 10)
    # ax1.set_xscale('log')

    # ax1.set_ylabel(r'emu/real - 1')
    ax1.set_ylim(-2e-4, 2e-4)


    ax0.plot(l, Cls.T, alpha = 0.3)



    plt.savefig('Plots/ClEmuPrior.png', figsize= (24,18), bbox_inches="tight", dpi = 900)
    plt.show()


#### POSTERIOR DRAWS ######

# plt.rcParams['axes.color_cycle'] = [ 'navy', 'forestgreen', 'darkred', 'sienna']


plt.rc('text', usetex=True)   # Slower
plt.rc('font', size=12)  # 18 usually

plt.figure(999, figsize=(7, 6))
from matplotlib import gridspec

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(hspace=0.02, left=0.2, bottom=0.15)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_ylabel(r'$C_l$', fontsize = 15)

ax1.axhline(y=0, ls='dashed')
# ax1.axhline(y=-1e-6, ls='dashed')
# ax1.axhline(y=1e-6, ls='dashed')

ax1.set_xlabel(r'$l$', fontsize = 15)


ax0.set_yscale('log', base = 10)
ax0.set_xscale('log', base = 10)
ax1.set_xscale('log', base = 10)

ax1.set_ylabel(r'emu/real - 1')
ax1.set_ylim(-5e-4, 5e-4)



ax0.plot(l, 10**Cls.T, alpha = 0.03, color = 'k')


color_id = 0
for x_id in del_idx:
    color_id = color_id + 1
#
# for x_id in range(0, num_holdout):

    time0 = time.time()
    x_decodedGPy = GP_predict(parameter_array[x_id])  ## input parameters
    time1 = time.time()
    print('Time per emulation %0.2f'% (time1 - time0), ' s')

    ax0.plot(l, 10**x_decodedGPy, alpha=1.0, ls='--', label='emu', color=plt.cm.Set1(color_id))


    time0 = time.time()
    x_decoded_new = GPy_predict(np.expand_dims(parameter_array[x_id], axis=0))

    time1 = time.time()
    print('Time per emulation %0.2f'% (time1 - time0), ' s')

    ax0.plot(l, 10**x_decoded_new, alpha=1.0, ls='--', label='emu', color=plt.cm.Set1(color_id))


    x_test = Cls[x_id]
    ax0.plot(l, 10**x_test, alpha=0.9, label='real', color=plt.cm.Set1(color_id))

    ax1.plot( l,  (10**x_decodedGPy)/(10**x_test) - 1, color=plt.cm.Set1(color_id))
    ax1.plot( l,  (10**x_decoded_new)/(10**x_test) - 1, ls='--', color=plt.cm.Set1(color_id))


    plt.legend()




    # ax1.plot(  (x_decodedGPy[1:])/(x_test[1:]) - 1)

ax0.set_xlim(l[0], l[-1])
ax1.set_xlim(l[0], l[-1])

ax0.set_xticklabels([])


plt.savefig('Plots/ClEmu.png', figsize= (28,24), bbox_inches="tight", dpi = 900)
plt.show()




#### Plot PCA bases and weights ####


PlotPCA = False

if PlotPCA:


    PCAbases = r('svd_decomp2$v[,1:nrankmax]')

    PCAweights = r('svd_weights2')



    plt.figure(900, figsize=(8,6))

    plt.title('Truncated PCA weights')
    plt.xlabel('PCA weight [0]',fontsize = 18)
    plt.ylabel('PCA weight [1]',fontsize = 18)
    CS = plt.scatter(PCAweights[:, 0], PCAweights[:, 1], c = parameter_array[num_holdout:, 2], s = 200, alpha=0.8)
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel(r'$\sigma_8$', fontsize = 18)
    plt.tight_layout()
    plt.savefig('Plots/SVD_TruncatedWeights8.png', figsize= (28,24), bbox_inches="tight", dpi = 900)



    plt.figure(901, figsize=(8,6))

    plt.title('Truncated PCA weights')
    plt.xlabel('PCA weight [0]',fontsize = 18)
    plt.ylabel('PCA weight [1]',fontsize = 18)
    CS = plt.scatter(PCAweights[:, 0], PCAweights[:, 1], c = parameter_array[num_holdout:, 5], s = 200, alpha=0.8)
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel(r'$z_m$', fontsize = 18)
    plt.tight_layout()
    plt.savefig('Plots/SVD_TruncatedWeightZm.png', figsize= (28,24), bbox_inches="tight", dpi = 900)





    plt.figure(902, figsize=(7,3))

    # plt.xlabel('PCA weight [0]',fontsize = 18)
    # plt.ylabel('PCA weight [1]',fontsize = 18)


    n_lines = 16
    # fig, ax = plt.subplots()
    plt.title('Truncated PCA bases')

    cmap = plt.cm.get_cmap('jet', n_lines)

    for i in range(n_lines):

        fig = plt.plot(PCAbases[:150,i], lw = 1.0, alpha = 0.8)

    # cbar = plt.colorbar(fig)
    # cbar.ax.set_ylabel(r'$z_m$', fontsize = 18)
    plt.tight_layout()
    plt.savefig('Plots/Bases.png', figsize= (28,24), bbox_inches="tight", dpi = 900)



######### TEMPLATE FOR MCMC LIKELIHOOD FUNCTION #######################
# For emcee

def lnlike(theta, x, y, yerr):
    p1, p2, p3, p4, p5 = theta

    new_params = np.array([p1, p2, p3, p4, p5])

    model = GP_predict(new_params)
    # return -0.5 * (np.sum(((y - model) / yerr) ** 2.))
    return -0.5 * (np.sum(((y - model) / yerr) ** 2.))


