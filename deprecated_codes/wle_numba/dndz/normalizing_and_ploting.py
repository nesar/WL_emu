import sys
import numpy as np
from scipy.interpolate import interp1d
import pylab as pl

def total_area(x_in,y_in):
	x_down = x_in[:-1]
	x_up = x_in[1:]
	y_down = y_in[:-1]
	y_up = y_in[1:]

	area_array = (y_down+y_up)*(x_up-x_down)/2.0

	return np.sum(area_array)


def cal_pdz(z_in):

	zs,dist = np.loadtxt("./source_distribution.txt",comments='#',usecols=(0,1),unpack=True)
	dist_normal = dist/total_area(zs,dist)
	f1 = interp1d(zs, dist_normal, kind='cubic')

	return f1(z_in)

if __name__ == '__main__':
    zsa = np.linspace(0.1,1.9,100)
    pdz = cal_pdz(zsa) # Probability of sources at a given redshift z

    pl.xlabel(r"$z_s$")
    pl.ylabel(r"Probability(z)")
    pl.plot(zsa,pdz)
    pl.show()
