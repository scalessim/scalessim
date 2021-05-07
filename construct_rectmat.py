import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
import scipy.ndimage
from astropy import units as u
import time

from scalessim.DFT import *
from scalessim.binning import *
from scalessim.phasescreen import *
from scalessim.pupil import *
from scalessim.io import *
from scalessim.focal_plane import *
from scalessim.targs import *
from scalessim.base import *
from scalessim.helpers import *

import configparser
from detector import nghxrg as ng

config = configparser.ConfigParser()
config.read('h2rg.ini')

arg_detector = {}
arg_detector.update(read_ini(config['Defined']))

ng_h2rg = ng.HXRGNoise(verbose=False,
                        wind_mode='WINDOW',
                        naxis1=1024, naxis2=1024,
                        pca0_file='./detector/lmircam_pca0.fits')

rd_noise=arg_detector['readout_noise_cds']*1.   # White read noise per integration
pedestal=arg_detector['pedestal']*1.   # DC pedestal drift rms
c_pink=arg_detector['c_pink']*1.     # Correlated pink noise
u_pink=arg_detector['u_pink']*1.     # Uncorrelated pink noise
acn=arg_detector['acn']*1.        # Correlated ACN
pca0_amp=arg_detector['pca0_amp']*1.   # Amplitude of PCA zero "picture frame" noise

#arg_detector



config.read('scales_h2rg.ini')
arg_spaxel = {}
arg_spaxel.update(read_ini(config['Defined']))
arg_spaxel.update(read_ini(config['Derived']))
arg_spaxel.update(read_ini(config['User']))

wav_min = 2.0
wav_max = 5.2
arg_spaxel['min_wavelength'] = wav_min #minimum wavelength in microns
arg_spaxel['max_wavelength'] = wav_max #maximum wavelength in microns

lens = Lenslet(arg_spaxel)
lens.get_shifts()
lens.make_trace(phys=True,disp=True,verbose=True)

vapor = 1 #PWV in mm
airmass = 1
skybg = SkyBG(vapor,airmass)
skytrans = SkyTrans(vapor,airmass)
atmodisp = AtmoDispersion(90,20,600)
inst_emissivities = [.08]*3 + [.01]*8
inst_temps = [273*u.K]*11
inst = InstTransEm(inst_emissivities, inst_temps)
qe = QE()
filt = Filter(lmin=lens.lmin,lmax=lens.lmax)
dit = 1 ###integration time in seconds

####organize all these and pass to focal_plane
args_new = {'Lenslet_object':lens,
            'SkyBG':skybg,
            'SkyTrans':skytrans,
            'InstTransEm':inst,
            'Filter':filt,
            'QE':qe}
fp = FocalPlane(args_new)



from scipy import sparse
def gen_rectmat(llens,linds):
    rectmat = []

    poss = llens.args['spaxel_size_px'] * np.arange(fp.num_spaxel)
    cinds = np.where(np.abs(fp.lam.value - llens.lmin)<1.0e-6)[0][0]
    cinde = np.where(np.abs(fp.lam.value - llens.lmax)<1.0e-6)[0][0]
    lamsx = fp.lam[cinds:cinde+1]
    toextr=llens.trace.copy()[cinds:cinde+1]
    bes = np.linspace(wav_min,wav_max,55)
    lams = 0.5*(bes[0:-1]+bes[1:])
    toextr_bin = np.zeros([54,toextr.shape[1],toextr.shape[2]])
    for xx in range(len(lamsx)):
        lam = lamsx[xx].value
        for yy in range(len(bes)-1):
            if lam > bes[yy]:
                if lam < bes[yy+1]:
                    toextr_bin[yy]+=toextr[xx]
    xloc = llens.xx[cinds]
    yloc = llens.yy[cinds]
    for ll in linds:
        start = 1
        print(ll)
        tinp = toextr_bin[ll]
        tosamp = np.zeros([2048,2048])
        tosamp[:len(tinp),:len(tinp[0])] = np.array(tinp)
        t2=0
        t1=0
        for ii in range(108):
            print(ll,ii,(t2-t1))
            t1 = time.time()

            for jj in range(108):
                sdx = poss[ii]-xloc
                sdy = poss[jj]-yloc
                toadd2 = shift(tosamp,(sdy,sdx),order=1,prefilter=False)
                ones = np.zeros(toadd2.shape)
                ones[np.where(toadd2 > np.max(toadd2)*0.25)] = 1.0
                #plt.imshow(ones)
                #plt.show()
                #if jj==2: stop
                toapp = ones.flatten()
                toapp = toapp.reshape([1,len(toapp)])
                if start == 1:
                    tmp = sparse.bsr_matrix(toapp)
                    start = 0
                else:
                    tmp = sparse.vstack([tmp,toapp])
            print(tmp.shape)
            t2 = time.time()

        print(tmp.shape)
        sparse.save_npz('rectmats/2.0_5.2_rectmat_'+str(ll)+'.npz',tmp)
    return tmp

lstart = int(sys.argv[1])
lstop = int(sys.argv[2])
linds = range(lstart,lstop)
gen_rectmat(lens,linds)
