'''
Created on Jun 29, 2017

'''
import numpy as np
import subprocess as sp
from numba import jit
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import pylab as pl

import source_redshift_dist as srd
import pade_funcs as pade

#----------------------------------------------------------------------
# Define the cosmology and constants used in this code.
#
vc = 2.998e5 #(km/s), speed of light
H0 = 71.0 #km/s / Mpc, Hubble constant
h = 0.71
Om0= 0.264 # Omega_M
ncosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

pk_dir = "./des_output_pk/" # the directory to save *_pk.dat
#----------------------------------------------------------------------
# calculate P(k,z) in $nbins$ redshift bins between zmin and zmax, and
# save the results in pk_dir. please see "./readme.txt" to learn
# how to run emu.exe to calculate P(k,z)
#
def main(zmin,zmax,nbins):
    dzl = (zmax-zmin)/nbins
    zl_array = np.linspace(zmin,zmax-dzl,nbins)+dzl/2.0 
    for i in xrange(nbins):
        filename = pk_dir+str(i)+"_"+str(zl_array[i])+"_"+str(dzl)+"_pk.dat"
        
        cmd = "./emu.exe "+filename +" 0.02258  0.1330824 0.963 71 -1.0 0.8 "+str(zl_array[i])
        sp.call(cmd,shell=True)

    return 0
#----------------------------------------------------------------------
# Extrapolation for the 2d Cl(l, zl)
#
def extrap1d(interpolator, xs, ys):
    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))

    return ufunclike
#----------------------------------------------------------------------
# Weighting function for a mass sheet at zl, and the redshift
# distribution of sources is  "./dndz/source_distribution.txt".
#

def wfunc_all(zl):
    zs_list = np.linspace(0.0001,1.9999,20)
    dzs = zs_list[1]-zs_list[0]
    total_wfunc = 0.0

    for zs in zs_list:
        if (zs<zl):
            continue

        Dc_l = ncosmo.comoving_distance(zl).value
        Dc_s = ncosmo.comoving_distance(zs).value
        Dc_ls = Dc_s - Dc_l

        tmp = dzs*srd.cal_pdz(zs)*Dc_ls/Dc_s
        total_wfunc = total_wfunc + tmp

    return total_wfunc
#----------------------------------------------------------------------
# Calculate 2d projected angular power spectrum Cl(l, zl),
# please see the Equation 29 in the paper:
# Martin Kilbinger (2015)
# http://iopscience.iop.org/article/10.1088/0034-4885/78/8/086901/pdf
#

def pk_2d(k,pk,z_lower,z_upper,z1):

    afactor = 1.0/(1.0+z1)
    cfactor = (3.0/2.0*(H0/vc)**2.0*Om0)

    Dc_l = ncosmo.comoving_distance(z1).value
    lk = k*Dc_l

    Dc_d = ncosmo.comoving_distance(z_lower).value
    Dc_u = ncosmo.comoving_distance(z_upper).value
    DDc = Dc_u - Dc_d 

    wfunc = wfunc_all(z1) 
    pkl = DDc*(cfactor*wfunc/afactor)**2.0*pk
    
    return lk,pkl
#----------------------------------------------------------------------
# Convert k to l according to the redshifts of lens planes,
# you may not need it in this code.
#
def l_to_k(l,z1):
    Dc_l = ncosmo.comoving_distance(z1).value
    res = l/Dc_l
    return res
#----------------------------------------------------------------------
@jit(nopython=True)
def numb_extrap1d(x, xs, ys):
    ptr=0
    y=x*0.0
    
    for i in xrange(len(x)):
        if x[i]<xs[0]:
            y[i]=ys[0]+(x[i]-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        else:
            while (ptr!=len(xs)) & (x[i]>=xs[ptr]):
                ptr=ptr+1
            if ptr==len(xs):
                y[i]=ys[-1]+(x[i]-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])  
            else:
                y[i]=ys[ptr]+ (x[i]-xs[ptr])*(ys[ptr]-ys[ptr-1])/(xs[ptr]-xs[ptr-1])  
    
    return y
#----------------------------------------------------------------------
# Integrate the projected 2d Cl(l, zl) along line of sight
# to obtain the efficient Cl(l).
#
def sum_pk_2d(k_array,pk_array,zl_array,dzl):
    l = np.linspace(0.01,1e4,1e5) 
    # l = 10.0**np.linspace(-2,5,1e4)
    res = l*0.0
    
    for i in xrange(len(zl_array)): 
        lktmp,pkltmp = pk_2d(k_array[i],pk_array[i],zl_array[i]-dzl/2,zl_array[i]+dzl/2,zl_array[i]) 
        ftmp = numb_extrap1d(np.log10(l), np.log10(lktmp),np.log10(pkltmp))
        res = res + 10**ftmp #Cl(l,zl) integn step to Cl(l) EQN. 29
    return l,res
#--------------------------------------------------------------------------
# Read in the 3d matter power spectrum from pl_dir and calculate
# the final Cl(l)
#
def multiple_zs():
    cmd = "ls " + pk_dir
    files = sp.check_output(cmd,shell=True)
    file_list = files.split("\n")[:-1] 
    zl_array = []
    pk_array = []
    lk_array = []
    
    #Each redshift will have a different approximant
    for i in xrange(len(file_list)):
        filename = pk_dir+file_list[i]
        k_tmp,pk_tmp = np.loadtxt(filename, dtype="double", usecols=(0,1),unpack=True)
        pts=[k_tmp[480],k_tmp[521], k_tmp[564]]
        rhs=[pk_tmp[480],pk_tmp[521], pk_tmp[564]]
        p,q=pade.pade_coeffs(3,0,2, pts, rhs)
        k_ext=np.linspace(8.5692+0.034, 250, 500)
        pk_ext=pade.n_point_pade(k_ext, p, q)
        k_tmp=np.append(k_tmp, k_ext)
        pk_tmp=np.append(pk_tmp, pk_ext)
        
        zl_tmp = np.double(file_list[i].split('_')[1]) 
        dzl = np.double(file_list[i].split('_')[2]) 
        
        zl_array.append(zl_tmp)
        lk_array.append(k_tmp)
        pk_array.append(pk_tmp)
    l, pkf = sum_pk_2d(lk_array,pk_array,zl_array,dzl)
    return l, pkf
#-----------------------------------------------------------------------
def multiple_zs_PL1(): #First Power Law scheme
    cmd = "ls " + pk_dir
    files = sp.check_output(cmd,shell=True)
    file_list = files.split("\n")[:-1] #Stores the names of files in pk_dir
    zl_array = []
    pk_array = []
    lk_array = []
    #Each redshift will have a different approximant
    for i in xrange(len(file_list)): #Each i is a different z
        filename = pk_dir+file_list[i]
        k, pk = np.loadtxt(filename, dtype="double", usecols=(0,1),unpack=True)
        k_tmp, pk_tmp=pade.pk_pl1(k, pk)
        
        zl_tmp = np.double(file_list[i].split('_')[1]) #str(zl_array[i]) mid point of the bin
        dzl = np.double(file_list[i].split('_')[2]) #str(dzl) width of the bin

        zl_array.append(zl_tmp)
        lk_array.append(k_tmp) #lk=[#z values*#k values array]
        pk_array.append(pk_tmp)

    l, pkf = sum_pk_2d(lk_array,pk_array,zl_array,dzl)

    return l, pkf

#-----------------------------------------------------------------------

def multiple_zs_PL2(): #Second Power Law scheme
    cmd = "ls " + pk_dir
    files = sp.check_output(cmd,shell=True)
    file_list = files.split("\n")[:-1] #Stores the names of files in pk_dir
    zl_array = []
    pk_array = []
    lk_array = []
    #Each redshift will have a different approximant
    for i in xrange(len(file_list)): #Each i is a different z
        filename = pk_dir+file_list[i]
        k, pk = np.loadtxt(filename, dtype="double", usecols=(0,1),unpack=True)
        k_tmp, pk_tmp=pade.pk_pl2(k, pk)
        
        zl_tmp = np.double(file_list[i].split('_')[1]) #str(zl_array[i]) mid point of the bin
        dzl = np.double(file_list[i].split('_')[2]) #str(dzl) width of the bin

        zl_array.append(zl_tmp)
        lk_array.append(k_tmp) #lk=[#z values*#k values array]
        pk_array.append(pk_tmp)

    l, pkf = sum_pk_2d(lk_array,pk_array,zl_array,dzl)

    return l, pkf

#-----------------------------------------------------------------------
# Calculate the 2d projected Cl(l, zl, zs) of each lens plane
# in the case of single source plane
#
def pk_2d_single(k,pk,z_lower,z_upper,z1,z2):

    afactor = 1.0/(1.0+z1)
    cfactor = (3.0/2.0*(H0/vc)**2.0*Om0)

    Dc_l = ncosmo.comoving_distance(z1).value
    lk = k*Dc_l

    Dc_d = ncosmo.comoving_distance(z_lower).value
    Dc_u = ncosmo.comoving_distance(z_upper).value
    DDc = Dc_u - Dc_d

    # if (z2<z1):
        # continue

    Dc_l = ncosmo.comoving_distance(z1).value
    Dc_s = ncosmo.comoving_distance(z2).value
    Dc_ls = Dc_s - Dc_l

    pkl = DDc*(cfactor*Dc_ls/Dc_s/afactor)**2.0*pk

    return lk,pkl
#-----------------------------------------------------------------------
# Integrate Cl(l, zl, zs) along line of sight to obtain
# the final Cl(l) in the case of single source plane
#
def sum_pk_2d_single(k_array,pk_array,zl_array,dzl,zs):
    # l = np.linspace(0.01,1e4,1e5)
    l = 10.0**np.linspace(-2,5,1e4)
    res = l*0.0

    for i in xrange(len(zl_array)):
        lktmp,pkltmp = pk_2d_single(k_array[i],pk_array[i],zl_array[i]-dzl/2,zl_array[i]+dzl/2,zl_array[i], zs)
        f_i = interp1d(np.log10(lktmp),np.log10(pkltmp),kind='linear')
        ftmp = extrap1d(f_i,np.log10(lktmp),np.log10(pkltmp))
        res = res + 10**ftmp(np.log10(l))

    return l,res
#----------------------------------------------------------------------
# Read in the 3d matter power spectrum from pl_dir and calculate
# the final Cl(l) in the case of single source plane
#
def single_zs(z2):
    cmd = "ls " + pk_dir
    files = sp.check_output(cmd,shell=True)
    file_list = files.split("\n")[:-1]
    zl_array = []
    pk_array = []
    lk_array = []

    for i in xrange(len(file_list)):
        filename = pk_dir+file_list[i]
        k_tmp,pk_tmp = np.loadtxt(filename, dtype="double", usecols=(0,1),unpack=True)
        
        zl_tmp = np.double(file_list[i].split('_')[1]) 
        dzl = np.double(file_list[i].split('_')[2]) 

        zl_array.append(zl_tmp)
        lk_array.append(k_tmp)
        pk_array.append(pk_tmp)

    l,pkf = sum_pk_2d(lk_array,pk_array,zl_array,dzl,z2)

    return l, pkf


if __name__ == '__main__':
#----------------------------------------------------------------------
# First, call main function to calculate 3d matter power spectrum,
# and save them to $pk_dir$. This function only need to call once,
# then you can comment it.
#
# main(0.0,2.0,100) #z ranges from 0 to 2 and has 100 bins

#----------------------------------------------------------------------
# Second, call multiple_zs() to integrate the powerspectrum along line of sight.
# Simply, this function will 1) convert 3d matter power spectrum to 2d angular
# powerspectrum; 2) integrate the 2d angular powerspectrum according to the
# redshift distribution of sources and lenses planes.
#
#    

    lf_pade, Clf_pade=multiple_zs()
    #lf_pl1, Clf_pl1=multiple_zs_PL1()
    #lf_pl2, Clf_pl2=multiple_zs_PL2()
    

    #np.savetxt('Input_0_pade.txt', np.c_[lf_pade, Clf_pade])
    #np.savetxt('Input_0_pl1.txt', np.c_[lf_pl1, Clf_pl1])
    #np.savetxt('Input_0_pl2.txt', np.c_[lf_pl2, Clf_pl2])
    
    pl.figure()
    pl.xlabel(r"$l (k \times D_{cl})$")
    pl.ylabel(r"$P_{2D}(l)$")
    pl.xlim(1e0,1e4)
    pl.ylim(1e-12,1e-7)
    pl.loglog(lf_pade, Clf_pade, "k-",label="Emulator")
    pl.title("Cl Extrapolation Comparisons")
    pl.legend()
    pl.show()
