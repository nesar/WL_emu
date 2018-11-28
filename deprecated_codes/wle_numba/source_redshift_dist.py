import numpy as np
from scipy.interpolate import interp1d

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

    return pointwise

def total_area(x_in,y_in):
    x_down = x_in[:-1]
    x_up = x_in[1:]
    y_down = y_in[:-1]
    y_up = y_in[1:]

    area_array = (y_down+y_up)*(x_up-x_down)/2.0

    return np.sum(area_array)


def cal_pdz(z_in):

    zs,dist1 = np.loadtxt("./dndz/source_distribution.txt",comments='#',usecols=(0,1),unpack=True)
    dist1_normal = dist1/total_area(zs,dist1)

    # np.savetxt("./srd_normed.dat",np.transpose(np.array([zs,dist1_normal])),fmt="%.6e")

    f1 = interp1d(zs, dist1_normal, kind='linear')
    f2 = extrap1d(f1,zs,dist1_normal)

    return f2(z_in)

# if __name__ == '__main__':
    # import pylab as pl
    # zs_list = np.linspace(0.02,1.9,20)
    # dzs = zs_list[1]-zs_list[0]
    # cal_pdz(z_in)
