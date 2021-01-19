import os
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
from .io import Prism
from scipy.ndimage import shift,filters

class Lenslet:
    def __init__(self, args):
        self.args = args
        self.n = args['n']
        self.fnum = args['lenslet_fnum']
        self.num = args['no_spaxel']
        self.l_pitch = args['spaxel_size']
        self.f = self.l_pitch * self.fnum
        self.p_pitch = args['px_pitch']
        self.spectra_l = args['spectra_length']
        self.spectra_sep = args['spectra_sep']
        self.lmin = args['min_wavelength']
        self.lmax = args['max_wavelength']

    def get_shifts(self, Prism=Prism()):
        self.Prism = Prism
        toscaley = self.spectra_l/(self.lmax-self.lmin)*(np.max(self.Prism.x)-np.min(self.Prism.x))*np.cos(np.radians(18.43))
        toscalex = self.spectra_l/(self.lmax-self.lmin)*(np.max(self.Prism.x)-np.min(self.Prism.x))*np.sin(np.radians(18.43))


        self.Prism.scale_to_length(toscaley.value)
        self.yy = self.Prism.y.copy()

        self.Prism.scale_to_length(toscalex.value)
        self.xx = self.Prism.y.copy()

        self.yy = self.yy-self.yy.mean()
        self.xx = self.xx-self.xx.mean()
        y1=(self.yy[np.where((np.abs(self.Prism.x.value-self.lmax)<1.0e-6))])
        y2=(self.yy[np.where((np.abs(self.Prism.x.value-self.lmin)<1.0e-6))])
        x1=(self.xx[np.where((np.abs(self.Prism.x.value-self.lmax)<1.0e-6))])
        x2=(self.xx[np.where((np.abs(self.Prism.x.value-self.lmin)<1.0e-6))])


    def make_trace(self, upsample_factor=100,phys=False,physdir='POPtxtFiles/SquarePrism45mm-rotated/',verbose=False):

        toutfile = 'data/traces/trace_h2rg_'+str(np.round(self.lmin,2))+'_'+str(np.round(self.lmax,2))+'.fits'
        if os.path.isfile(toutfile)==False:
            mshift = np.max([np.abs(self.xx),np.abs(self.yy)])
            osize = int(np.round(2.0*mshift))+2
            if osize%2 != 0: osize+=1
            out_size = (osize,osize)
            out = np.zeros((len(self.xx), *out_size))
            y_screen = np.linspace(-out_size[0]//2,out_size[0]//2,out_size[0]*upsample_factor)[:,None] * self.p_pitch
            x_screen = np.linspace(-out_size[1]//2,out_size[1]//2,out_size[1]*upsample_factor)[None,:] * self.p_pitch
            for i, (x, y, lam) in enumerate(zip(self.xx, self.yy, self.Prism.x)):
                if verbose==True: print(i,lam)
                y_screen2 = y_screen - y*self.p_pitch
                x_screen2 = x_screen - x*self.p_pitch
                del_y = np.pi * self.l_pitch * np.sin(np.arctan2(y_screen2, self.f))
                del_x = np.pi * self.l_pitch * np.sin(np.arctan2(x_screen2, self.f))
                temp = np.sinc((del_y / lam.value / np.pi).value)**2 * np.sinc((del_x / lam.value / np.pi).value)**2 * (np.pi)**2
                ######I divide by pi inside the sinc and multiply by pi^2 because numpy's sinc is sin(pi x)/(pi x)
                temp /= np.sum(temp) / upsample_factor**2
                if phys == True:
                    temp = pyfits.getdata(physdir+'54micronPinhole'+str(np.round(lam.value,4))+'micronPSF_s'+str(self.p_pitch/upsample_factor)+'um.fits')
                ####the following line just bins down by upsample factor
                pixels = out_size[0]
                sample = temp.reshape((pixels, upsample_factor, pixels, upsample_factor)).mean(3).mean(1)
                shifted = sample
                ####this shifts the image to go to the correct position in the trace
                #padded = np.pad(sample, int((out_size[0]-pixels)/2), mode='minimum')
                #shifted = shift(padded,[y,x],prefilter=False)

                out[i] = shifted

            self.trace = out
            plt.imshow(np.sum(self.trace,axis=0)**0.1)
            plt.show()
            pyfits.writeto(toutfile,np.array(self.trace))
        else: self.trace = pyfits.getdata(toutfile)
