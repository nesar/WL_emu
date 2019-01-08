'''
Created on Jun 29, 2017

'''

import sys
sys.path.append('/home/nes/Desktop/AstroVAE/WL_emu/Codes/deprecated_codes/wle_numba/')
sys.path.append('/Users/tricia/Documents/Work_files/WL_Emulator/deprecated_codes/wle_numba/')
import numpy as np
import subprocess as sp
from numba import jit
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import pylab as pl

import source_redshift_dist as srd
import pade_funcs as pade
from scipy.integrate import fixed_quad


#----------------------------------------------------------------------
# Define the cosmology and constants used in this code.
#
vc = 2.998e5 #(km/s), speed of light

pk_dir = "./pk_outputs_g/"
#pk_dir = "./des_output_pk/" # the directory to save *_pk.dat
#----------------------------------------------------------------------
# calculate P(k,z) in $nbins$ redshift bins between zmin and zmax, and
# save the results in pk_dir. please see "./readme.txt" to learn
# how to run emu.exe to calculate P(k,z)
#



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
# weighting functions

def pdz_given(z,zm,fwhm):
    return 1./(np.sqrt(2*np.pi) * (fwhm/2.0)) * np.exp(-2.0*(z-zm)**2/fwhm**2)

def pdz_arbitrary():
    ''' Code to allow arbitrary redshift distribution '''
    nz = np.loadtxt('nz.txt')[:,1]
    z = np.loadtxt('nz.txt')[:,0]
    pdz = interp1d(z,nz,bounds_error=False,fill_value=0.0)
    return pdz

def wfunc_int(zs,chil,pdz,zl,fwhm,ncosmo):
    chis = ncosmo.comoving_distance(zs).value
    return pdz(zs,zl,fwhm)*(chis - chil)/chis


def wfunc_int_arb(zs,chil,pdz,ncosmo):
    chis = ncosmo.comoving_distance(zs).value
    return pdz(zs)*(chis - chil)/chis


def wfunc_fixedquad(zl,nval,pdz,zm,fwhm,ncosmo):
    Dc_l = ncosmo.comoving_distance(zl).value
    w,werr = fixed_quad(wfunc_int,zl,2.0,args=(Dc_l,pdz,zm,fwhm,ncosmo),n=nval)
    return w,werr

def wfunc_fixedquad_arbitrary(zl,nval,pdz,ncosmo):
    Dc_l = ncosmo.comoving_distance(zl).value
    w,werr = fixed_quad(wfunc_int_arb,zl,2.0,args=(Dc_l,pdz,ncosmo),n=nval)
    return w,werr


#----------------------------------------------------------------------
# Calculate 2d projected angular power spectrum Cl(l, zl),
# please see the Equation 29 in the paper:
# Martin Kilbinger (2015)
# http://iopscience.iop.org/article/10.1088/0034-4885/78/8/086901/pdf
#

def pk_2d(k,pk,z_lower,z_upper,z1,cfactor,zm,fwhm,input_nz,ncosmo):

    afactor = 1.0/(1.0+z1)

    Dc_l = ncosmo.comoving_distance(z1).value
    lk = k*Dc_l

    Dc_d = ncosmo.comoving_distance(z_lower).value
    Dc_u = ncosmo.comoving_distance(z_upper).value
    DDc = Dc_u - Dc_d 
 
    #pdz = cal_pdz_new()
    if input_nz:
        pdz = pdz_arbitrary();
        wfunc = wfunc_fixedquad_arbitrary(z1,100,pdz,ncosmo)[0]
    else:
        wfunc = wfunc_fixedquad(z1,100,pdz_given,zm,fwhm,ncosmo)[0]    
    #wfunc = wfunc_fixedquad(z1,100,pdz,zm,fwhm,ncosmo)[0]
    #wfunc = wfunc_all(z1) 
    pkl = DDc*(cfactor*wfunc/afactor)**2.0*pk
    
    return lk,pkl
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
def sum_pk_2d(k_array,pk_array,zl_array,dzl,cfactor,zm,fwhm,input_nz,ncosmo):
    l = np.arange(10000) 
    res = l*0.0
    for i in xrange(len(zl_array)): 
        lktmp,pkltmp = pk_2d(k_array[i],pk_array[i],zl_array[i]-dzl/2,zl_array[i]+dzl/2,zl_array[i],cfactor,zm,fwhm,input_nz,ncosmo) 
        ftmp = numb_extrap1d(np.log10(l), np.log10(lktmp),np.log10(pkltmp))
        res = res + 10**ftmp #Cl(l,zl) integn step to Cl(l) EQN. 29
    return l,res
#--------------------------------------------------------------------------


# Read in the 3d matter power spectrum from pl_dir and calculate
# the final Cl(l)
#
def multiple_zs(Om0,H0,zm=1.0,fwhm=0.5,input_nz=False):
    cmd = "ls " + pk_dir
    files = sp.check_output(cmd,shell=True)
    file_list = files.split("\n")[:-1] 
    zl_array = []
    pk_array = []
    lk_array = []
    cfactor = (3.0/2.0*(H0/vc)**2.0*Om0)
    ncosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

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
        #print(file_list[i])
        #print(dzl)       
        zl_array.append(zl_tmp)
        lk_array.append(k_tmp)
        pk_array.append(pk_tmp)
    #print(dzl)
    l, pkf = sum_pk_2d(lk_array,pk_array,zl_array,dzl,cfactor,zm,fwhm,input_nz,ncosmo)
    return l, pkf
#-----------------------------------------------------------------------


if __name__ == '__main__':
#----------------------------------------------------------------------
# Second, call multiple_zs() to integrate the powerspectrum along line of sight.
# Simply, this function will 1) convert 3d matter power spectrum to 2d angular
# powerspectrum; 2) integrate the 2d angular powerspectrum according to the
# redshift distribution of sources and lenses planes.
#
#    
    print("STOP STOP STOP")
    lf_pade, Clf_pade=multiple_zs()
       
