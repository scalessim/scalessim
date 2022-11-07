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

        #Comment this out to see if we actually need it; it's defined within get_fp
        self.dlam = self.Prism.get_dlam()
        self.lam = self.Prism.ll
        #Need lmin and lmax to trim PSF cube
        self.lmin = args['min_wavelength']
        self.lmax = args['max_wavelength']
        #self.pre_image = self.Target.preimage

        self.buffer = self.Lenslet.trace.shape[1] // 2

        self.area = self.Lenslet.args['area'] * u.m**2#np.pi*((self.Lenslet.args['telescope_diameter']/2)**2 - (self.Lenslet.args['secondary_diameter']/2)**2) * u.m**2

    def get_fp(self, dit, Target=None, PSF=None, bg_off=False, cube=None, return_full=True, verbose=False, return_phots=False, medium=False, bra=False):
        output = np.zeros((len(self.lam), self.num_spaxel, self.num_spaxel))
        if medium==True and bra==False:
            output = np.zeros((len(self.lam), self.num_spaxel-1, self.num_spaxel))

        if medium==True and bra==True:
            ###Add code to create smaller PSF cube if bra=True
            minll2 = np.argmin(np.abs(self.Prism.ll - self.lmin * u.micron))
            maxll2 = np.argmin(np.abs(self.Prism.ll - self.lmax * u.micron))
            self.Prism.ll2 = self.Prism.ll[minll2:maxll2]
            self.dlam = self.dlam[minll2:maxll2]
            self.lam = self.Prism.ll2
            self.xx2 = self.Lenslet.xx[minll2:maxll2] - np.min(self.Lenslet.xx) + 10.0
            self.yy2 = self.Lenslet.yy[minll2:maxll2] - np.min(self.Lenslet.yy) + 10.0
            output = np.zeros((len(self.lam), self.num_spaxel-1, self.num_spaxel))


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

            #Should probably do an else in case of low resolution mode
            if medium==True and bra==True:
                xsize=len(self.xx2)
                ysize=len(self.yy2)
                dx = self.xx2[0]
                dy = self.yy2[0]
                dx_max = self.xx2[-1]
                dy_max = self.yy2[-1]

            else:
                y1=(self.Lenslet.yy2[np.where((np.abs(self.lam.value-lmax)<1.0e-6))])[0]
                y2=(self.Lenslet.yy2[np.where((np.abs(self.lam.value-lmin)<1.0e-6))])[0]
                x1=(self.Lenslet.xx2[np.where((np.abs(self.lam.value-lmax)<1.0e-6))])[0]
                x2=(self.Lenslet.xx2[np.where((np.abs(self.lam.value-lmin)<1.0e-6))])[0]
                xsize=(x1-x2)
                ysize=(y1-y2)
                cind = np.where(np.abs(self.lam.value - lmin)<1.0e-6)[0][0]
                cind_max = np.where(np.abs(self.lam.value - lmax)<1.0e-6)[0][0]
                dx = self.Lenslet.xx2[cind]
                dy = self.Lenslet.yy2[cind]
                dx_max = self.Lenslet.xx2[cind_max]
                dy_max = self.Lenslet.yy2[cind_max]

            dims = int(np.ceil(self.Lenslet.args['spaxel_size_px']*(self.num_spaxel - 1) + xsize))
            poss = self.Lenslet.args['spaxel_size_px'] * np.arange(self.num_spaxel)

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
            print('nspax = ',self.num_spaxel)


            if medium==True:
                #shifts = pyfits.getdata('/Users/stephsallum/Dropbox/scalessim/data/medres_shifts.fits')
                #shifts = pyfits.getdata('/Users/ram/scalessim/data/medres_shifts.fits')
                #More realistic shifts for placement on detector
                shifts = pyfits.getdata('/Users/ram/scalessim/data/medres_shifts2.fits')
                for ii in range(self.num_spaxel):
                #for ii in range(1,2):
                    if verbose==True: print(ii)
                    for jj in range(self.num_spaxel-1):
                        if verbose == True: print(ii,jj)
                        shiftx,shifty = shifts[ii,jj,1],shifts[ii,jj,0]

                        tinp = self.Lenslet.trace.copy()*img[:,jj,ii].reshape([len(img),1,1])
                        tinp = tinp.sum(0)



                        toadd = np.zeros([tinp.shape[0]+2,tinp.shape[1]+2])
                        dxt = len(toadd[0])
                        dyt = len(toadd)
                        toadd[1:1+len(tinp),1:1+len(tinp[0])] = np.array(tinp)


                        ###shifty is where the spectrum should get moved to
                        ####dy is where the spectrum starts in the trace
                        dimy = shifty-int(shifty) - (dy-int(dy))
                        dimx = shiftx-int(shiftx) - (dx-int(dx))
                        ###shift by the correct non-integer value
                        print(dimx,dimy)
                        toadd = shift(toadd,(dimy,dimx),order=1,prefilter=False)

                        ####crop to correct amount
                        pad = 6
                        toadd = toadd[int(dy-pad):int(dy+ysize+pad),int(dx-pad):int(dx+xsize+pad)]
                        print(toadd.shape)
                        ystart = int(shifty)
                        xstart = int(shiftx)
                        xend = int(shiftx)+len(toadd[0])
                        yend = int(shifty)+len(toadd)
                        print(xstart,ystart)
                        print(xend,yend)
                        if xend > 2048:
                            toadd = toadd[:,:-(xend-2048)]
                            xend = 2048
                        if yend > 2048:
                            toadd = toadd[:-(yend-2048),:]
                        """
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
                        print(imy,imx)
                        print(toadd.shape)
                        stop
                        """
                        pyfits.writeto('test_toadd.fits',np.array(toadd),clobber=True)
                        out_array[ystart:yend,xstart:xend] += toadd
            else:
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





