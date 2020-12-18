import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits

#=================sky model=====================
class PhaseScreen(object):
    '''generate phase screen
    method:
        get_res
    '''
    def __init__(self, r0=0.1, L0=None, pixsize=None,
        sizem=None, pupil = None, strehl = None):
        '''
        input:
            None
        keyword:
            (float) r0 = 0.1: fred paramener in meter
            (float) L0 = None: outer scale in meter, None for infinity
            (float) pixsize = None: output pixel size of the phase screen
            (float) sizem = None: phyical size of the phase screen
            (bool) pupil = None: pupil object, pixsize and sizem will copy from pupil if it is set.
            (float) strehl = None: re-scale RMS of the phase due to the stehl ratio of the image.
        '''
        self.pixsize = pixsize
        self.sizem = sizem

        self.r0 = r0
        self.uncorelated = True
        self.L0 = L0
        self.ref = 500e-9
        self.sr_factor = np.nan

        if pupil:
            self.pixsize = pupil.pupil.shape[0]
            self.sizem = self.pixsize * pupil.rate

        if strehl:
            self.sr_factor = self.set_strehl(strehl['value'], strehl['wl'])

        self.get_filter()

    def set_strehl(self, sr, wl):
        factor = -np.log(sr)
        factor = np.sqrt(factor) * wl / self.ref
        return factor

    def get_strehl(self, phase, wl):
        sr = ((phase - phase.mean())**2 * (self.ref/wl)**2).mean()
        return np.exp(-sr)

    def dist(self,pixsize):
        nx = np.arange(pixsize)-pixsize/2
        gxx,gyy = np.meshgrid(nx,nx)
        freq = gxx**2 + gyy**2
        freq = np.sqrt(freq)
        return np.fft.ifftshift(freq)

    def get_filter(self):
        freq = self.dist(self.pixsize)/self.sizem
        freq[0,0] = 1.0
        factors = np.sqrt(0.00058)*self.r0**(-5.0/6.0)
        factors *= np.sqrt(2)*2*np.pi/self.sizem

        if not self.L0:
            self.filter = factors * freq**(-11.0/6.0)
        else:
            self.filter = factors * (freq ** 2 + self.L0**(-2))**(-11.0/12.0)

        self.filter[0,0] = 0

    def get_res(self):
        '''return the phase screen.
        output:
            phase screen
        '''
        if self.uncorelated:
            return self.new_phs_long_enough()

    def new_phs_long_enough(self):
        phase = np.random.randn(self.pixsize,self.pixsize)*np.pi
        x_phase = np.cos(phase) + 1j*np.sin(phase)
        pscreen = np.fft.ifft2(x_phase*self.filter)
        ps = np.real(pscreen)*self.pixsize**2

        if np.isnan(self.sr_factor):
            return ps
        else:
            return ps/ps.std()*self.sr_factor
