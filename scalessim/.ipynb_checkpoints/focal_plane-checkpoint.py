from scipy.ndimage import shift
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from ipywidgets import IntProgress,Label
from IPython.display import display
from copy import deepcopy

class FocalPlane:
    def __init__(self, args, gain=8):

        self.gain = gain #e/DN
        self.Lenslet = args['Lenslet_object']
        self.SkyBG = args['SkyBG']
        self.SkyTrans = args['SkyTrans']
        self.Inst = args['InstTransEm']
        self.QE = args['QE']
        self.Filter = args['Filter']
        self.fov = (self.Lenslet.args['fov']*u.arcsec)**2
        self.num_spaxel = self.Lenslet.num

        self.Prism = self.Lenslet.Prism
        self.dlam = self.Prism.get_dlam()
        self.lam = self.Prism.ll


        self.area = self.Lenslet.args['area'] * u.m**2

    def get_fp(self, dit, Target=None, PSF=None, bg_off=False, cube=None, return_full=True,verbose=False,return_phots=False,medium=False):
        output = np.zeros((len(self.lam), self.num_spaxel, self.num_spaxel))
        if medium==True:
            output = np.zeros((len(self.lam), self.num_spaxel-1, self.num_spaxel))
            

        skybg = self.SkyBG.resample(self.lam) * self.fov / self.num_spaxel**2
        instbg = self.Inst.get_em(self.lam) * self.fov / self.num_spaxel**2
        qe = self.QE.get_qe(self.lam)
        
        
        self.bg = skybg + instbg

        filtertrans = self.Filter.interp(self.lam)
        skytrans = self.SkyTrans.resample(self.lam)
        teltrans,insttrans = self.Inst.get_trans(self.lam)

        
        
        self.trans = teltrans*insttrans*filtertrans*skytrans

        
        
        bg_spec_in_phot = dit*(teltrans*insttrans*filtertrans*skybg + insttrans*filtertrans*instbg) * self.dlam * self.area.to(u.cm**2)
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
                #print('scaling cube')

        if return_full:
            lmin=self.Lenslet.args['min_wavelength']
            lmax=self.Lenslet.args['max_wavelength']
            
            poss = self.Lenslet.args['spaxel_size_px'] * np.arange(self.num_spaxel)
            
            
            cind = np.where(np.abs(self.lam.value - lmin)<1.0e-6)[0][0]
            cind_max = np.where(np.abs(self.lam.value - lmax)<1.0e-6)[0][0]

            
            
            testx,testy = self.Lenslet.xx2[0],self.Lenslet.yy2[0]
            offsetsx = 28.0 - (self.Lenslet.xx2 - testx)
            offsetsy = 28.0 - (self.Lenslet.yy2 - testy)


            
            
            dx = self.Lenslet.xx2[cind]
            dy = self.Lenslet.yy2[cind]

            
            dxm = self.Lenslet.xx2[cind_max]
            dym = self.Lenslet.yy2[cind_max]
            
            npix=self.Lenslet.args['detector_px']
            out_array = np.zeros([2048,2048])

            
            xtlocs = poss-offsetsx[cind]
            ytlocs = poss-offsetsy[cind]
            


            shiftsx = xtlocs - np.array(xtlocs,dtype='int')
            shiftsy = ytlocs - np.array(ytlocs,dtype='int')
            
            
            if medium==False:
                if verbose==True: 
                    f2 = IntProgress(min=0, max=108*108) # instantiate the bar
                    count = 0
                    label2 = Label(value="Progress: "+str(count)+' of '+str(108*108)+' lenslets')
                    display(label2)
                    display(f2)                    
                count=0
                for ii in range(self.num_spaxel):
                    sdx = xtlocs[ii]
                    for jj in range(self.num_spaxel):
                        count+=1
                        if count%108==0:
                            if verbose==True:
                                f2.value=count
                                label2.value = "Progress: "+str(count)+' of '+str(108*108)+' lenslets'
                        sdy = ytlocs[jj]

                        
                        tinp = self.Lenslet.trace.copy()*img[:,jj,ii].reshape([len(img),1,1])
                        tinp = tinp.sum(0)
                        if len(np.where(tinp < 0)[0]) > 0:
                            print('trace negative')
                            print(ii,jj)
                        if len(np.where(img[:,jj,ii] < 0)[0]) > 0:
                            print('input image negative')
                            print(ii,jj)
                            stop                        
                        toadd = np.zeros([tinp.shape[0]+2,tinp.shape[1]+2])
                        toadd[1:1+len(tinp),1:1+len(tinp[0])] = np.array(tinp)
                        

                        
                        dxt = len(toadd[0])
                        dyt = len(toadd)


                        imy = int(sdy)
                        imx = int(sdx)
                        dimy = sdy - imy
                        dimx = sdx - imx

                        toadd = shift(toadd,(dimy,dimx),order=1,prefilter=False)
                        if len(np.where(toadd < 0)[0]) > 0:
                            print('shifted image negative')
                            print(ii,jj)
                            stop
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
                        
                        
                        if len(np.where(np.isnan(out_array)==True)[0]!=0):
                            print(ii,jj)
                            print('oops, nan')
                            stop
                        if len(np.where(out_array<0)[0]!=0):
                            print(ii,jj)
                            print('oops, negative - thats weird!')
                            stop
                        
                if verbose==True:
                    f2.bar_style = 'success'
                    f2.close()
                    label2.close()
            
            
            

            
            if medium==True:
                if verbose==True: 
                    f2 = IntProgress(min=0, max=17*18) # instantiate the bar
                    count = 0
                    label2 = Label(value="Progress: "+str(count)+' of '+str(18*17)+' lenslets')
                    display(label2)
                    display(f2)   
                shifts = pyfits.getdata('data/medres_shifts_new.fits')
                shifts[:,:,0]+=10
                shifts[:,:,1]+=10
                
                

                
                for ii in range(self.num_spaxel):
                    for jj in range(self.num_spaxel-1):

                        
                    
                        count+=1
                        if verbose==True:
                            f2.value=count
                            label2.value = "Progress: "+str(count)+' of '+str(18*17)+' lenslets'
                        shiftx,shifty = shifts[ii,jj,1],shifts[ii,jj,0]

                    
                        ycst = int(dy-shifty)-1
                        ypl = 0
                        if ycst < 0:
                            ypl = -ycst
                            ycst = 0
                        yce = ycst+2050
                        
                        xcst = int(dx-shiftx)-1
                        xpl = 0
                        if xcst < 0:
                            xpl = -xcst
                            xcst = 0
                        xce = xcst+2050
                        
                    
                        im_mult = img[:,jj,ii].reshape([len(img),1,1])
                        tinp = self.Lenslet.trace[:,ycst:yce,xcst:xce]*im_mult
                        tinp = tinp.sum(0)
                        
                        ###shifty is where the spectrum should get moved to
                        ####dy is where the spectrum starts in the trace
                        dimy = shifty-int(shifty) - (dy-int(dy))
                        dimx = shiftx-int(shiftx) - (dx-int(dx))
                        
                        
                        
                        ###shift by the correct non-integer value
                        toadd = shift(tinp,(dimy,dimx),order=1,prefilter=False)

                      
                        xend = xpl + len(toadd[0])
                        yend = ypl + len(toadd)
                        
                        if xend > 2048:
                            toadd = toadd[:,:-(xend-2048)]
                            xend = 2048
                        if yend > 2048:
                            toadd = toadd[:-(yend-2048),:]
                            yend = 2048
            
                        out_array[ypl:yend,xpl:xend] += toadd

                if verbose==True:
                    f2.bar_style = 'success'
                    f2.close()
                    label2.close()
            
            return out_array
        else:
            return img





