import numpy as np
from scipy.interpolate import*

def pade_coeffs (n, l ,m ,xis, rhs): #n: number of points
    
    a_coeffs=[]
    b_coeffs=[]
    
    for i in xrange(0, n):
        a_coeffs.append([])
        for j in xrange(0, l+1):
            a_coeffs[i].append(xis[i]**j)
            
    
    for i in xrange(0,n):
        b_coeffs.append([])        
        
        for j in xrange(0, m):
            t=-(xis[i]**(j+1))*rhs[i]
            b_coeffs[i].append(t)
        
    a_coeffs=np.array(a_coeffs)
    b_coeffs=np.array(b_coeffs)

    A=np.array(np.concatenate((a_coeffs, b_coeffs), axis=1))
    soln=np.linalg.solve(A, rhs)
    pcs=soln[:l+1]
    qcs=soln[l+1:]
    p_c=np.flipud(pcs)
    q_c= np.concatenate(([1], qcs))
    return p_c, np.flipud(q_c)


def n_point_pade (x, p_c, q_c):
    return (np.poly1d(p_c)(x))/(np.poly1d(q_c)(x))


def pk_pl1(k, pk): #First Power Law scheme
    intp=interpolate.interp1d(np.log(k[480:565]), np.log(pk[480:565]), kind='linear')
    #print np.log(k[564])-np.log(k[521])
    #print np.log(k[480])-np.log(k[521])
    
    tay_p=approximate_taylor_polynomial(intp, np.log(k[521]), 1, 0.2)
    yo=(tay_p.c)[1]
    y_prime=(tay_p.c)[0]
    
    A,n=solve(np.log(k[521]), yo, y_prime)
    k_ext=np.linspace(8.5692+0.034, 250, 500)
    pk_ext=A*k_ext**n
    k_tmp=np.append(k, k_ext)
    pk_tmp=np.append(pk, pk_ext)
    return k_tmp, pk_tmp
    

def pk_pl2(k, pk): #Second Power Law scheme
    d=0.0
    
    for i in [x for x in xrange(518, 525) if x != 521]:
        diff=(np.log(pk[521])-np.log(pk[i]))/(np.log(k[521])-np.log(k[i]))
        d=d+diff

    y_prime=d/6
    y=np.log(pk[521])
    A,n=solve(np.log(k[521]), y, y_prime)
    k_ext=np.linspace(8.5692+0.034, 250, 500)
    pk_ext=A*k_ext**n
    k_tmp=np.append(k, k_ext)
    pk_tmp=np.append(pk, pk_ext)
    return k_tmp, pk_tmp
    
def solve(xo, y, y_diff):
    n=y_diff
    A=np.exp(y-(y_diff*xo))
    return A,n
