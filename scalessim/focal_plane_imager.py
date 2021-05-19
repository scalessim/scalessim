from scipy.ndimage import shift
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

class FocalPlaneImager:
    def __init__(self, args, gain=8):

        self.gain = gain #e/DN
        self.SkyBG = args['SkyBG']
        self.SkyTrans = args['SkyTrans']
        #elf.AtmoDispersion = args['AtmoDispersion']
        self.Inst = args['InstTransEm']
        self.QE = args['QE']
        self.Filter = args['Filter']
        ltmp = self.Filter.x[np.where(self.Filter.x > np.min(self.SkyBG.x))]
        ltmp2 = ltmp[np.where(ltmp < np.max(self.SkyBG.x))]

        dlamtmp = ltmp2[1:]-ltmp2[:-1]
        self.lam = ltmp2[:-1]+0.5*dlamtmp
        self.dlam = dlamtmp

        #self.dlam = self.lam[1]-self.lam[0] ####assumes wavelengths are evenly spaced!


        self.fov = args['FOV'] * u.arcsec**2
        #self.platescale = args['PlateScale']
        self.area = args['area'] * u.m**2
        self.npix = args['DetectorPix']




    def get_fp(self, dit, Target=None, PSF=None, bg_off=False, cube=None, return_full=True,verbose=False,return_phots=False):
        output = np.zeros((self.npix,self.npix))
        skybg = self.SkyBG.resample(self.lam) * self.fov / self.npix**2
        instbg = self.Inst.get_em(self.lam) * self.fov / self.npix**2
        qe = self.QE.get_qe(self.lam)

        self.bg = skybg + instbg

        filtertrans = self.Filter.interp(self.lam)
        skytrans = self.SkyTrans.resample(self.lam)
        teltrans,insttrans = self.Inst.get_trans(self.lam)

        self.trans = teltrans*insttrans*filtertrans*skytrans

        bg_spec_in_phot = dit*(teltrans*insttrans*filtertrans*skybg + insttrans*filtertrans*instbg) * self.dlam * self.area.to(u.cm**2)
        bg_spec_in_dn = bg_spec_in_phot*qe / self.gain / u.electron


        img = np.ones_like(output)

        if not bg_off:
            if return_phots == True:
                img = img * np.sum(bg_spec_in_phot[:,None,None].value)
            else:
                img = img * np.sum(bg_spec_in_dn[:, None, None].si.value)

        if Target:
            source = Target.resample(self.lam)
            h = 6.621e-27*u.cm*u.cm*u.g/u.s
            c = 2.9979e10*u.cm/u.s
            lamscm = self.lam.to(u.cm)
            source2 = source.to(u.cm*u.cm*u.g/u.s/u.s/u.cm/u.cm/u.micron/u.s) * lamscm / h / c * u.ph
            source_spec_in_phot = dit*self.trans*source2 * self.dlam * self.area.to(u.cm**2)
            source_spec_in_dn = source_spec_in_phot*qe / self.gain / u.electron

            for lll in range(len(PSF)):
                if return_phots ==True:
                    img += PSF[lll] * source_spec_in_phot[lll].value
                else:
                    img += PSF[lll] * source_spec_in_dn[lll].si.value



        return img





