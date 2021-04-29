import os
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
from .io import Prism
from scipy.ndimage import shift,filters,zoom

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

    def get_shifts(self, rot = 18.43):
        self.Prism = Prism(self.lmin,self.lmax)
        self.rot = rot

        #####the following lines use the default, fixed rotation angle
        #toscaley = self.spectra_l/(self.lmax-self.lmin)*(np.max(self.Prism.x)-np.min(self.Prism.x))*np.cos(np.radians(18.43))
        #toscalex = self.spectra_l/(self.lmax-self.lmin)*(np.max(self.Prism.x)-np.min(self.Prism.x))*np.sin(np.radians(18.43))
        ####these are rescaling, but we don't need to do this with the real dispersion curves
        #toscaley = self.spectra_l/(self.lmax-self.lmin)*(np.max(self.Prism.x)-np.min(self.Prism.x))*np.cos(np.radians(self.rot))
        #toscalex = self.spectra_l/(self.lmax-self.lmin)*(np.max(self.Prism.x)-np.min(self.Prism.x))*np.sin(np.radians(self.rot))
        #self.Prism.scale_to_length(toscaley.value)
        #self.yy = self.Prism.y.copy()
        #self.Prism.scale_to_length(toscalex.value)
        #self.xx = self.Prism.y.copy()

        #self.yy = self.yy-self.yy.mean()
        #self.xx = self.xx-self.xx.mean()
        #y1=(self.yy[np.where((np.abs(self.Prism.x.value-self.lmax)<1.0e-6))])
        #y2=(self.yy[np.where((np.abs(self.Prism.x.value-self.lmin)<1.0e-6))])
        #x1=(self.xx[np.where((np.abs(self.Prism.x.value-self.lmax)<1.0e-6))])
        #x2=(self.xx[np.where((np.abs(self.Prism.x.value-self.lmin)<1.0e-6))])


        self.xx = self.Prism.x/self.p_pitch
        self.yy = self.Prism.y/self.p_pitch

    def make_trace(self, upsample_factor=100,disp = False, phys=False,physdir='POPtxtFiles/SquarePrism45mm-rotated/',verbose=False):

        if disp==False:
            tbase = 'data/traces/trace_h2rg_'
        else:
            tbase = 'data/traces_disp/trace_h2rg_'

        if phys==True: tbase+='POP_'
        toutfile = tbase+str(np.round(self.lmin,2))+'_'+str(np.round(self.lmax,2))+'_'+str(self.rot)+'.fits'
        if os.path.isfile(toutfile)==False:
            #self.xx2 = self.xx - self.xx[int(len(self.xx)/2)]
            #self.yy2 = self.yy - self.yy[int(len(self.yy)/2)]
            self.xx2 = self.xx - np.min(self.xx) + 10.0
            self.yy2 = self.yy - np.min(self.yy) + 10.0
            mshifty = np.max(np.abs(self.yy2))
            mshiftx = np.max(np.abs(self.xx2))
            plt.scatter(self.xx2,self.yy2)
            plt.show()
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
                if phys == False:
                    #y_screen2 = y_screen - y*self.p_pitch ###swap in if we want to skip the shifting later
                    #x_screen2 = x_screen - x*self.p_pitch
                    #y_screen2 = y_screen
                    #x_screen2 = x_screen

                    del_y = np.pi * self.l_pitch * np.sin(np.arctan2(y_screen2, self.f))
                    del_x = np.pi * self.l_pitch * np.sin(np.arctan2(x_screen2, self.f))
                    temp = np.sinc((del_y / lam.value / np.pi))**2 * np.sinc((del_x / lam.value / np.pi))**2 * (np.pi)**2

                    #temp = np.sinc((del_y / lam.value / np.pi).value)**2 * np.sinc((del_x / lam.value / np.pi).value)**2 * (np.pi)**2 ###uncomment if we want to skip the shifting later
                    ######I divide by pi inside the sinc and multiply by pi^2 because numpy's sinc is sin(pi x)/(pi x)

                    pixels = out_size[0]
                    sample = temp.reshape((pixels, upsample_factor, pixels, upsample_factor)).mean(3).mean(1)
                    shifted = sample

                    ####this shifts the image to go to the correct position in the trace
                    padded = np.pad(sample, int((out_size[0]-pixels)/2), mode='minimum')
                    shifted = shift(padded,[y,x],prefilter=False)


                    shifted = shifted/np.sum(shifted)
                if phys == True:
                    temp2 = pyfits.getdata(physdir+'54micronPinhole'+str(np.round(lam.value,4))+'micronPSF_s'+str(self.p_pitch/upsample_factor)+'um_rot.fits')
                    """
                    the following few lines deal with the fact that we don't have physical optics
                    psfs that bracket the whole wavelength range of SCALES right now - I'm interpolating
                    between Phil's psfs which are sampled from 2.0-5.0 microns. So I just take the 2.0
                    version and zoom out to make the shorter than 2.0 psfs, and zoom in on the 5.0 version
                    to make the longer than 5.0 psfs. will replace this when we get a couple more psfs
                    from Phil
                    """
                    if True in np.isnan(temp2):
                        temp2 = pyfits.getdata(physdir+'54micronPinhole'+str(np.round(lam.value))+'micronPSF_s'+str(self.p_pitch/upsample_factor)+'um_rot.fits')
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
                    padded = np.pad(temp2,((topady,topady),(topadx,topadx)),mode='minimum')
                    #padded /= np.sum(padded) / upsample_factor**2
                    #plt.imshow(padded)
                    #plt.show()


                    sample = padded.reshape((out_size[0],upsample_factor,out_size[1],upsample_factor)).mean(3).mean(1)
                    #f = plt.figure(figsize=(15,15))
                    #plt.imshow(sample**0.1)
                    #plt.show()


                    #print(out.shape)

                    #print(y,x)

                    dy = sample.shape[0]/2.0-y
                    dx = sample.shape[1]/2.0-x

                    #print(dy,dx)

                    shifted = shift(sample,[dy,dx],prefilter=False)
                    shifted = shifted/np.sum(shifted)
                    #f = plt.figure(figsize=(15,15))
                    #plt.imshow(shifted**0.1)
                    #plt.show()
                    #stop

                out[i] += shifted

            #print(out.shape)
            #print(np.sum(np.sum(out,axis=-1),axis=-1))
            self.trace = out
            pyfits.writeto(toutfile,np.array(self.trace),clobber=True)
        else: self.trace = pyfits.getdata(toutfile)
        #plt.imshow(np.sum(self.trace,axis=0)**0.1)
        #plt.show()
