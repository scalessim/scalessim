import numpy as np
from .io import read_ini
from .base import Lenslet
from detector import nghxrg as ng
import configparser


#lets model detector noise

config = configparser.ConfigParser()
config.read('h2rg.ini')


arg_detector = {}
arg_detector.update(read_ini(config['Defined']))

ng_h2rg = ng.HXRGNoise(verbose=False,
						wind_mode='WINDOW',
						naxis1=1024, naxis2=1024,
						pca0_file='./detector/lmircam_pca0.fits')

# Use parameters that generate noise similar to LMIRCam
rd_noise=arg_detector['readout_noise_cds']*1.   # White read noise per integration
pedestal=arg_detector['pedestal']*1.   # DC pedestal drift rms
c_pink=arg_detector['c_pink']*1.     # Correlated pink noise
u_pink=arg_detector['u_pink']*1.     # Uncorrelated pink noise
acn=arg_detector['acn']*1.        # Correlated ACN
pca0_amp=arg_detector['pca0_amp']*1.   # Amplitude of PCA zero "picture frame" noise

# my_hdu = ng_h2rg.mknoise(None,rd_noise=rd_noise, pedestal=pedestal,
#                 c_pink=c_pink, u_pink=u_pink, acn=acn, pca0_amp=pca0_amp)

#lets model some spectral traces
config.read('scales_h2rg.ini')

arg_spaxel = {}
arg_spaxel.update(read_ini(config['Defined']))
arg_spaxel.update(read_ini(config['Derived']))
arg_spaxel.update(read_ini(config['User']))


lens = Lenslet(arg_spaxel)
lens.get_shifts()
trace = lens.make_trace(phys=False)
