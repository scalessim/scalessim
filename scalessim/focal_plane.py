from scipy.ndimage import shift
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

class FocalPlane:
    def __init__(self, args, gain=8):

        self.gain = gain #e/DN
        self.Lenslet = args['Lenslet_object']
        #self.Target = args['Target']
        self.SkyBG = args['SkyBG']
        self.SkyTrans = args['SkyTrans']
        #elf.AtmoDispersion = args['AtmoDispersion']
        self.Inst = args['InstTransEm']
        self.QE = args['QE']
        self.Prism = self.Lenslet.Prism
        self.Filter = args['Filter']
        self.fov = (self.Lenslet.args['fov']*u.arcsec)**2
        self.num_spaxel = self.Lenslet.num

        self.dlam = self.Prism.get_dlam()
        self.lam = self.Prism.ll
        #self.pre_image = self.Target.preimage

        self.buffer = self.Lenslet.trace.shape[1] // 2

        self.area = self.Lenslet.args['area'] * u.m**2#np.pi*((self.Lenslet.args['telescope_diameter']/2)**2 - (self.Lenslet.args['secondary_diameter']/2)**2) * u.m**2

    def get_fp(self, dit, Target=None, PSF=None, bg_off=False, cube=None, return_full=True,verbose=False,return_phots=False):
        output = np.zeros((len(self.lam), self.num_spaxel, self.num_spaxel))


        #test = self.SkyBG.resample(self.lam)
        #plt.plot(self.SkyBG.x,self.SkyBG.y)
        #plt.plot(self.lam,test)
        #plt.show()
        #print(test)

        #print(self.fov)
        #print(self.num_spaxel)
        #stop

        skybg = self.SkyBG.resample(self.lam) * self.fov / self.num_spaxel**2

        instbg = self.Inst.get_em(self.lam) * self.fov / self.num_spaxel**2
        qe = self.QE.get_qe(self.lam)

        self.bg = skybg + instbg

        filtertrans = self.Filter.interp(self.lam)
        skytrans = self.SkyTrans.resample(self.lam)
        teltrans,insttrans = self.Inst.get_trans(self.lam)
        #print(skybg)
        #stop
        #print(instbg)
        #print(teltrans)
        #print(insttrans)

        self.trans = teltrans*insttrans*filtertrans*skytrans

        bg_spec_in_phot = dit*(teltrans*insttrans*filtertrans*skybg + insttrans*filtertrans*instbg) * self.dlam * self.area.to(u.cm**2)
        #print(bg_spec_in_phot)
        #stop

        #bg_spec_in_phot = dit*insttrans*filtertrans*(skybg + instbg) * self.dlam * self.area.to(u.cm**2)
        bg_spec_in_dn = bg_spec_in_phot*qe / self.gain / u.electron



        if not bg_off:
            img = np.ones_like(output)
            if return_phots == True:
                img = img * bg_spec_in_phot[:,None,None].value
            else:
                img = img * bg_spec_in_dn[:, None, None].si.value
        else:
            img = np.zeros(output.shape)
        if Target:
            source = Target.resample(self.lam)
            h = 6.621e-27*u.cm*u.cm*u.g/u.s
            c = 2.9979e10*u.cm/u.s
            lamscm = self.lam.to(u.cm)
            source2 = source.to(u.cm*u.cm*u.g/u.s/u.s/u.cm/u.cm/u.micron/u.s) * lamscm / h / c * u.ph
            source_spec_in_phot = dit*self.trans*source2 * self.dlam * self.area.to(u.cm**2)
            source_spec_in_dn = source_spec_in_phot*qe / self.gain / u.electron
            print(source_spec_in_dn)


            if return_phots == True:
                img += PSF * source_spec_in_phot[:, None, None].value

            else:
                img += PSF * source_spec_in_dn[:, None, None].si.value

        if cube is not None:
            print('using cube')
            h = 6.621e-27*u.cm*u.cm*u.g/u.s
            c = 2.9979e10*u.cm/u.s
            lamscm = self.lam.to(u.cm)
            cube2 = []
            for x in range(len(cube)):
                tmp = cube[x].to(u.cm*u.cm*u.g/u.s/u.s/u.cm/u.cm/u.micron/u.s) * lamscm[x] / h / c * u.ph
                cube2.append(tmp)
            mult_phot = dit*self.trans * self.dlam * self.area.to(u.cm**2)
            mult_dn = mult_phot * qe / self.gain / u.electron

            if return_phots == True:
                img += (cube2 * mult_phot[:, None, None]).value
            else:
                img += (cube2 * mult_dn[:, None, None]).value


        if return_full:
            print('making full raw image')
            lmin=self.Lenslet.args['min_wavelength']
            lmax=self.Lenslet.args['max_wavelength']

            y1=(self.Lenslet.yy[np.where((np.abs(self.lam.value-lmax)<1.0e-6))])[0]
            y2=(self.Lenslet.yy[np.where((np.abs(self.lam.value-lmin)<1.0e-6))])[0]
            x1=(self.Lenslet.xx[np.where((np.abs(self.lam.value-lmax)<1.0e-6))])[0]
            x2=(self.Lenslet.xx[np.where((np.abs(self.lam.value-lmin)<1.0e-6))])[0]
            xsize = (x1-x2)
            ysize=(y1-y2)
            dims = int(np.ceil(self.Lenslet.args['spaxel_size_px']*(self.num_spaxel - 1) + xsize))
            poss = self.Lenslet.args['spaxel_size_px'] * np.arange(self.num_spaxel)

            cind = np.where(np.abs(self.lam.value - lmin)<1.0e-6)[0][0]
            dx = self.Lenslet.xx[cind]
            dy = self.Lenslet.yy[cind]

            #npx = self.Lenslet.trace[0].shape[1]/2
            #npy = self.Lenslet.trace[0].shape[0]/2
            npx = 0
            npy = 0

            xloc=(npx+dx)
            yloc=(npy+dy)
            npix=self.Lenslet.args['detector_px']
            distx=self.Lenslet.args['spectra_sep'] #centroid trace separation in pixels
            disty=self.Lenslet.spectra_l*np.cos(np.arctan(1.0/self.Lenslet.n))

            ddx = self.Lenslet.spectra_sep/ \
                    np.sin(np.arctan(1.0/self.Lenslet.n))
            ddy = ddx
            out_array = np.zeros([2048,2048])

            xtlocs = poss-xloc
            ytlocs = poss-yloc

            shiftsx = xtlocs - np.array(xtlocs,dtype='int')
            shiftsy = ytlocs - np.array(ytlocs,dtype='int')

            for ii in range(self.num_spaxel):
                if verbose==True: print(ii)
                sdx = xtlocs[ii]
                for jj in range(self.num_spaxel):
                    sdy = ytlocs[jj]
                    tinp = self.Lenslet.trace.copy()*img[:,jj,ii].reshape([len(img),1,1])
                    tinp = tinp.sum(0)

                    toadd = np.zeros([tinp.shape[0]+2,tinp.shape[1]+2])
                    dxt = len(toadd[0])
                    dyt = len(toadd)
                    toadd[1:1+len(tinp),1:1+len(tinp[0])] = np.array(tinp)


                    imy = int(sdy)
                    imx = int(sdx)
                    dimy = sdy - imy
                    dimx = sdx - imx

                    toadd = shift(toadd,(dimy,dimx),order=1,prefilter=False)
                    imy = imy - 1
                    imx = imx - 1
                    if imy < 0:
                        toadd = toadd[int(np.abs(imy)):,:]
                        py = 0
                        dyt = len(toadd)
                        imy = 0
                    if imx < 0:
                        toadd = toadd[:,int(np.abs(imx)):]
                        px = 0
                        dxt = len(toadd[0])
                        imx = 0
                    lpx = imx+dxt
                    if lpx > 2048:
                        cx = lpx-2048
                        toadd = toadd[:,:-cx]
                        dxt = len(toadd[0])
                    lpy = imy+dyt
                    if lpy > 2048:
                        cy = lpy-2048
                        toadd = toadd[:-cy,:]
                        dyt = len(toadd)
                    out_array[imy:imy+dyt,imx:imx+dxt] += toadd
            return out_array, img
        else:
            return img





