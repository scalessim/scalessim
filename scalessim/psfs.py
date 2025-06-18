import astropy.io.fits as pyfits
import numpy as np
from scipy.ndimage import zoom, shift
import matplotlib.pyplot as plt

class PSFs:
    def __init__(self,guidestar,scales):

        if scales.sim_med == True: self.med=True
        else: self.med=False
        print(self.med)
        self.datadir = 'data/PSFs/'
        
        all_psfs = []
        all_psfs_c = []
        Hmag_sim = int(guidestar.value)

        all_psfs = pyfits.getdata(self.datadir+'L_current_psfs_120nm_mag'+str(Hmag_sim)+'.fits')[:,1:109,1:109]
        all_psfs_c = pyfits.getdata(self.datadir+'L_current_coros_120nm_mag'+str(Hmag_sim)+'.fits')[:,1:109,1:109]
        
        self.all_psfs = all_psfs
        self.all_psfs_c = all_psfs_c

    def interp_psfs_wav(self, psf, lamc=3.4):
        lams_binned=np.linspace(1.9,5.3,341)
        if self.med == True:
            lams_binned=np.linspace(1.9,5.3,34001)
        psf_cube = []
        for i in range(len(lams_binned)):
            tmp = zoom(psf,lams_binned[i]/lamc,order=1,grid_mode=False,prefilter=False)
            if len(tmp) <= 108:
                tmp2 = np.zeros([108,108])
                if len(tmp)%2!=0:
                    tmp2[54-len(tmp)//2:54+len(tmp)//2,54-len(tmp)//2:54+len(tmp)//2]=shift(tmp[:-1,:-1],(-0.5,-0.5),order=1)
                else:
                    tmp2[54-len(tmp)//2:54+len(tmp)//2,54-len(tmp)//2:54+len(tmp)//2]=tmp
            if len(tmp)>108:
                tmp2=tmp[len(tmp)//2-54:54+len(tmp)//2,len(tmp)//2-54:54+len(tmp)//2]
            if self.med==False:
                psf_cube.append(tmp2) 
            if self.med==True:
                psf_cube.append(tmp2[54-8:54+9,54-9:54+9])
        return psf_cube        

    def PSF_sequence(self, nframes = 1, verbose=False, vortex=False):
        psfs = []
        psfs_c = []
        if nframes > len(self.all_psfs):
            rep = True
        else: rep = False
        randints = np.random.choice(len(self.all_psfs),nframes,replace=rep)
        for x in range(nframes):
            psfs_sing = self.all_psfs[randints[x]]/np.sum(self.all_psfs[randints[x]])
            psf_cube = self.interp_psfs_wav(psfs_sing)
            psfs.append(psf_cube)
            if vortex == True:
                psfs_sing_c = self.all_psfs_c[randints[x]]/np.sum(self.all_psfs[randints[x]])
                psf_cube_c = self.interp_psfs_wav(psfs_sing_c)
                psfs_c.append(psf_cube_c)
        psfs = np.array(psfs)
        psfs_c = np.array(psfs_c)
        if vortex == False: return psfs
        if vortex == True: return psfs, psfs_c

    def convolve(self, psf, scene):
        scene_conv = np.zeros_like(psf)
        if len(psf.shape)==3:
            FT_psf=np.array([np.fft.fft2(psf[i]) for i in range(len(psf))])
            FT_scene=np.array([np.fft.fft2(scene[i]) for i in range(len(scene))])
            FT_conv=FT_psf*FT_scene
            for i in range(len(scene_conv)):
                scene_conv[i] = np.array([np.real(np.fft.ifftshift(np.fft.ifft2(FT_conv[jj]))) 
                                     for jj in range(len(FT_conv))])
        if len(psf.shape)>3:
            for i in range(len(psf)):
                FT_psf = np.array([np.fft.fft2(psf[i][jj]) for jj in range(len(psf[i]))])
                FT_scene = np.array([np.fft.fft2(scene[k]) for k in range(len(scene))])
                FT_conv = FT_psf*FT_scene
                scene_conv[i] = np.array([np.real(np.fft.ifftshift(np.fft.ifft2(FT_conv[jj]))) 
                                     for jj in range(len(FT_conv))])
        return scene_conv

    
