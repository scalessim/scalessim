import configparser
from .detector import nghxrg as ng
from .io import read_ini
import os

class Detector:
    def __init__(self):

        config = configparser.ConfigParser()
        config.read('scalessim/detector/h2rg.ini')

        self.arg_detector = {}
        self.arg_detector.update(read_ini(config['Defined']))
        
        self.ng_h2rg = ng.HXRGNoise(verbose=False,
                                wind_mode='WINDOW',
                                naxis1=2048, naxis2=2048,
                                pca0_file='scalessim/detector/lmircam_pca0.fits')


    def make_noise(self,ofile=None,rn=1.,ped=1.,cpink=1.,upink=1.,acn=1.,pca0_amp=1.):
        self.rd_noise=self.arg_detector['readout_noise_cds']*rn   # White read noise per integration
        self.pedestal=self.arg_detector['pedestal']*ped   # DC pedestal drift rms
        self.c_pink=self.arg_detector['c_pink']*cpink     # Correlated pink noise
        self.u_pink=self.arg_detector['u_pink']*upink     # Uncorrelated pink noise
        self.acn=self.arg_detector['acn']*acn        # Correlated ACN
        self.pca0_amp=self.arg_detector['pca0_amp']*pca0_amp   # Amplitude of PCA zero "picture frame" noise

        return self.ng_h2rg.mknoise(ofile)
