import os
import numpy as np
import scipy
from scipy import interpolate

home = '/Users/ram/scalessim/'

lminlist = [1.95,2.90,4.50]
lmaxlist = [2.45,4.15,5.20]
mreslist = [4275.0,2716.0,6673.0]#mid-band spectral resolution from PDF document (Table 4.10)
lmidlist = [2.20,3.525,4.85]

for i in range(len(lminlist)):
	lmin = lminlist[i]
	lmax = lmaxlist[i]
	mres = mreslist[i]
	lmid = lmidlist[i]

	dlam = lmid/mres
	bw = lmax - lmin
	npoints = bw/dlam + 1
	npoints = np.int(np.floor(npoints))
	print(npoints)
	#lams_binned = np.linspace(lmin,lmax,npoints)

	filename = str(lmin)+'_'+str(lmax)+'_grism.txt'
	print(home+'data/'+filename)
	#if os.path.isfile('./data/{}'.format(filename))==True:
	if os.path.isfile(home+'data/'+filename)==True:
#####need to add ys to OG prism files
		ll, x, y = np.loadtxt(home+'data/'+filename, unpack=True)
		print(y)
###units of dispersion curve x and y are mm!!!
		#lams_des = lams_binned=np.linspace(1.9,5.3,3401)
		#xinterp = interpolate.interp1d(ll,x,kind='cubic')
		#yinterp = interpolate.interp1d(ll,y,kind='cubic')
		#x2 = xinterp(lams_des)
		#y2 = yinterp(lams_des)
		#ll = lams_des*u.micron
		#x = x2*1000.0 ##Grism file is in microns
		#y = y2*1000.0 ##Grism file is in microns
		#print(y)
	else:
		print('no prism data!')