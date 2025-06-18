import os
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
from .io import Prism, Grating, read_ini
from scipy.ndimage import shift,filters,zoom
import configparser


class Lenslet:
    def __init__(self, medium=False):
        self.med = medium
        conffile = 'data/scales_h2rg.ini'
        if self.med==True: conffile='data/scales_h2rg_med.ini'
        config = configparser.ConfigParser()
        config.read(conffile)
        arg_spaxel = {}
        arg_spaxel.update(read_ini(config['Defined']))
        arg_spaxel.update(read_ini(config['Derived']))
        arg_spaxel.update(read_ini(config['User']))
        self.args = arg_spaxel

        self.n = self.args['n']
        self.fnum = self.args['lenslet_fnum']
        self.num = self.args['no_spaxel']
        self.l_pitch = self.args['spaxel_size']
        self.f = self.l_pitch * self.fnum
        self.p_pitch = self.args['px_pitch']
        self.spectra_l = self.args['spectra_length']
        self.spectra_sep = self.args['spectra_sep']
        self.lmin = self.args['min_wavelength']
        self.lmax = self.args['max_wavelength']

    def get_shifts(self, rot = 18.43):
        self.Prism = Prism(self.lmin,self.lmax)
        self.rot = rot

        if self.med==True:
            self.Prism = Grating(self.lmin,self.lmax)
            self.rot = 0.0

        self.xx = self.Prism.x/self.p_pitch
        self.yy = self.Prism.y/self.p_pitch
        self.xx2 = self.xx - np.min(self.xx) + 10.0
        self.yy2 = self.yy - np.min(self.yy) + 10.0

    def make_trace(self, upsample_factor=100,verbose=False):
        tbase = 'data/traces_disp/trace_h2rg_POP_'
        physdir = 'data/POPtxtFiles/SquarePrism45mm-rotated/'
        toutfile = tbase+str(np.round(self.lmin,2))+'_'+str(np.round(self.lmax,2))+'_'+str(self.rot)+'.fits'

        if self.med==True: 
            tbase+='med_'
            toutfile = tbase+str(np.round(self.lmin,2))+'_'+str(np.round(self.lmax,2))+'.fits'


        if os.path.isfile(toutfile)==False:
        
            self.xx2 = self.xx - np.min(self.xx) + 10.0
            self.yy2 = self.yy - np.min(self.yy) + 10.0
            mshifty = np.max(np.abs(self.yy2))
            mshiftx = np.max(np.abs(self.xx2))
            #plt.scatter(self.xx2,self.yy2)
            #plt.show()
            osizey = int(np.round(mshifty))+28
            osizex = int(np.round(mshiftx))+28
            if osizey%2 != 0: osizey+=1
            if osizex%2 != 0: osizex+=1
            out_size = (osizey,osizex)
            out = np.zeros((len(self.xx2), *out_size))
            print(out.shape)
            y_screen2 = np.linspace(-out_size[0]//2,out_size[0]//2,out_size[0]*upsample_factor)[:,None] * self.p_pitch
            x_screen2 = np.linspace(-out_size[1]//2,out_size[1]//2,out_size[1]*upsample_factor)[None,:] * self.p_pitch

            for i, (x, y, lam) in enumerate(zip(self.xx2, self.yy2, self.Prism.ll)):
                if verbose==True: print(i,lam)

                psffile = physdir+'54micronPinhole'+str(np.round(lam.value,4))+\
                        'micronPSF_s'+str(self.p_pitch/upsample_factor)+'um_rot.fits'

                if self.med==True: 
                    psffile = physdir+'54micronPinhole'+str(np.round(lam.value,4))+\
                        'micronPSF_s'+str(self.p_pitch/upsample_factor)+'um_med.fits'
                #print(psffile)
                #stop

                if os.path.isfile(psffile):
                    temp2 = pyfits.getdata(psffile)
                    no_im=False
                else:
                    no_im=True
                    
                
                
                """
                the following few lines deal with the fact that we don't have physical optics
                psfs that bracket the whole wavelength range of SCALES right now - I'm interpolating
                between Phil's psfs which are sampled from 2.0-5.0 microns. So I just take the 2.0
                version and zoom out to make the shorter than 2.0 psfs, and zoom in on the 5.0 version
                to make the longer than 5.0 psfs. will replace this when we get a couple more psfs
                from Phil
                """
                if ((True in np.isnan(temp2)) or (zoom==True)):
                    temp2 = pyfits.getdata(physdir+'54micronPinhole'+str(np.round(lam.value))+
                                           'micronPSF_s'+str(self.p_pitch/upsample_factor)+'um_rot.fits')
                    #print('replacing',np.round(lam.value))
                    temp2 = zoom(temp2,lam.value/np.round(lam.value))
                    if len(temp2)%2!=0: temp2 = temp2[1:,1:]

                pxy,pxx = temp2.shape

                #plt.imshow(temp2)
                #plt.show()

                topady = int((out_size[0]*upsample_factor-pxy)/2)
                topadx = int((out_size[1]*upsample_factor-pxx)/2)

                if topady<0:
                    temp2 = temp2[-topady:topady]
                    topady=0
                if topadx < 0:
                    temp2 = temp2[:,-topadx:topadx]
                    topadx = 0
                padded = np.pad(temp2,((topady,topady),(topadx,topadx)))


                sample = padded.reshape((out_size[0],upsample_factor,out_size[1],upsample_factor)).mean(3).mean(1)

                dy = sample.shape[0]/2.0-y
                dx = sample.shape[1]/2.0-x


                #print(sample.shape)
                #plt.imshow(sample)
                #plt.xlim(len(sample[0])/2-30,len(sample[0])/2+30)
                #plt.ylim(len(sample)/2-30,len(sample)/2+30)
                #plt.colorbar()
                #plt.show()
                
                
                shifted = shift(sample,[-dy,-dx],prefilter=False) ###changed these signs on 5/5!!!! now it's moving the image to y,x
                #shifted = shift(sample,[2,5],prefilter=False)
                shifted = shifted/np.sum(shifted)

                out[i] += shifted
                #print(y,x)
                #print(dy,dx)
                #plt.imshow(shifted)
                #plt.xlim(len(sample[0])/2-30,len(sample[0])/2+30)
                #plt.ylim(len(sample)/2-30,len(sample)/2+30)
                #plt.xlim(0,20)
                #plt.ylim(0,50)
                #plt.show()
                #stop

            self.trace = out
            pyfits.writeto(toutfile,np.array(self.trace),overwrite=True)
        else: self.trace = pyfits.getdata(toutfile)
