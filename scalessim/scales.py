from scalessim.focal_plane import *
from scalessim.base import *
from scalessim.io import *
from ipywidgets import IntProgress, Label
from IPython.display import display
from scalessim.detectors import *


class SCALES:
    def __init__(self,scalesmode,verbose=False):
        self.res = scalesmode.value.split(' ')[0]
        if self.res == 'Med-Res': 
            self.sim_med = True
            self.upsamp = 10
        else: 
            self.sim_med = False
            self.upsamp = 10
        self.lens = Lenslet(medium=self.sim_med)

        self.wav_min = float(scalesmode.value.split(':')[1].split('-')[0])
        self.wav_max = float(scalesmode.value.split(':')[1].split('-')[1])
        self.lens.min_wavelength = self.wav_min
        self.lens.max_wavelength = self.wav_max
        self.lens.args['min_wavelength'] = self.wav_min
        self.lens.args['max_wavelength'] = self.wav_max
        self.lens.lmin = self.wav_min
        self.lens.lmax = self.wav_max
        self.lens.get_shifts()
        self.lens.make_trace(verbose=verbose,upsample_factor=self.upsamp)
        self.filt = Filter(lmin=self.lens.lmin,lmax=self.lens.lmax,fkw='asahi')



    def image(self,targ=None,psf=None,cube_in=None,
             inst_emissivities = [0.4],inst_temps = [277*u.K],
             dit=1,shot_off=False,verbose=False,hxrg_off=False,
             skytransoff=False,bkg_off=False):


        vapor = 1.0 #PWV in mm
        airmass = 1.0 
        self.skybg = SkyBG(vapor,airmass)
        self.skytrans = SkyTrans(vapor,airmass)
        if skytransoff==True: self.skytrans.y = np.ones(self.skytrans.y.shape)*u.dimensionless_unscaled
        self.atmodisp = AtmoDispersion(90,20,600)

        self.inst = InstTransEm(inst_emissivities, inst_temps)
        self.qe = QE()
        #dit = 1 ###integration time in seconds


        self.args = {'Lenslet_object':self.lens,
                    'SkyBG':self.skybg,
                    'SkyTrans':self.skytrans,
                    'InstTransEm':self.inst,
                    'Filter':self.filt,
                    'QE':self.qe}

        self.fp = FocalPlane(self.args)


            
        if verbose==True:
            if targ!=None:
                if len(psf.shape)>3:
                    max_count = len(psf)
                else:
                    max_count = 1
            else:
                if len(cube_in.shape)>3:
                    max_count = len(cube_in)
                else:
                    max_count = 1
            count=0
            f = IntProgress(min=0, max=max_count) # instantiate the bar
            label = Label(value="Progress: "+str(count)+' of '+str(max_count)+' images')
            display(label)
            display(f)


        if targ!=None:
            if len(psf.shape)>3:
                raw_seq = []
                for iii in range(len(psf)):
                    raw = self.fp.get_fp(dit*u.s,targ,return_full=True,PSF=psf[iii],bg_off = bkg_off,verbose=verbose,medium=self.sim_med)
                    raw_seq.append(raw)
                    if verbose==True: 
                        count+=1
                        f.value = count
                        label.value="Progress: "+str(count)+' of '+str(max_count)+' images'

            else:
                raw_seq = [self.fp.get_fp(dit*u.s,targ,return_full=True,PSF=psf[iii],bg_off = bkg_off,verbose=verbose,medium=self.sim_med)]
                if verbose==True: 
                    count+=1
                    f.value = count
                    label.value="Progress: "+str(count)+' of '+str(max_count)+' images'

        else:
            if len(cube_in.shape)>3:
                raw_seq = []
                for iii in range(len(cube_in)):
                    raw = self.fp.get_fp(dit*u.s,cube=cube_in[iii],return_full=True,bg_off = bkg_off,verbose=verbose,medium=self.sim_med)
                    raw_seq.append(raw)
                    if verbose==True: 
                        count+=1
                        f.value = count
                        label.value="Progress: "+str(count)+' of '+str(max_count)+' images'

            else:
                raw_seq=[self.fp.get_fp(dit*u.s,cube=cube_in,return_full=True,bg_off = bkg_off,verbose=verbose,medium=self.sim_med)]
                if verbose==True: 
                    count+=1
                    f.value = count
                    label.value="Progress: "+str(count)+' of '+str(max_count)+' images'

                    
        if verbose==True:
            # Finish the progress bar
            f.bar_style = 'success'
            f.close()
            label.close()
        
        raw_seq=np.array(raw_seq)
        print(np.where(raw_seq < 0))
        if shot_off==False: raw_seq = np.random.poisson(raw_seq)
        if hxrg_off==False:
            detector = Detector()
            for i in range(len(raw_seq)):
                hdu = detector.make_noise()
                raw_seq[i]+=np.array(hdu.data,dtype='int')
        return raw_seq


    def cube(self,targ=None,psf=None,cube_in=None,
             inst_emissivities = [0.4],inst_temps = [277*u.K],
             dit=1,return_oversampled=False,shot_off=False,
             verbose=False,skytransoff=False,bkg_off=False,
             vapor = 1.0,airmass=1.0):

        self.skybg = SkyBG(vapor,airmass)
        self.skytrans = SkyTrans(vapor,airmass)
        if skytransoff==True: self.skytrans.y = np.ones(self.skytrans.y.shape)*u.dimensionless_unscaled

        self.atmodisp = AtmoDispersion(90,20,600)
        self.inst = InstTransEm(inst_emissivities, inst_temps)
        self.qe = QE()


        self.args = {'Lenslet_object':self.lens,
                    'SkyBG':self.skybg,
                    'SkyTrans':self.skytrans,
                    'InstTransEm':self.inst,
                    'Filter':self.filt,
                    'QE':self.qe}

        self.fp = FocalPlane(self.args)


        if verbose==True:
            if targ!=None:
                if len(psf.shape)>3:
                    max_count = len(psf)
                else:
                    max_count = 1
            else:
                if len(cube_in.shape)>3:
                    max_count = len(cube_in)
                else:
                    max_count = 1
            count=0
            f = IntProgress(min=0, max=max_count) # instantiate the bar
            label = Label(value="Progress: "+str(count)+' of '+str(max_count)+' images')
            display(label)
            display(f)

        
        
        if targ!=None:
            if len(psf.shape)>3:
                cube_seq = []
                for iii in range(len(psf)):
                    cube_out = self.fp.get_fp(dit*u.s,Target=targ,return_full=False,PSF=psf[iii],bg_off = bkg_off, verbose=verbose,medium=self.sim_med)
                    cube_seq.append(cube_out)
                    if verbose==True: 
                        count+=1
                        f.value = count
            else:
                cube_seq=[self.fp.get_fp(dit*u.s,Target=targ,return_full=False,PSF=psf,bg_off = bkg_off,verbose=verbose,medium=self.sim_med)]
                if verbose==True: 
                    count+=1
                    f.value = count
        
        else:
            if len(cube_in.shape)>3:
                cube_seq = []
                for iii in range(len(cube_in)):
                    cube_out = self.fp.get_fp(dit*u.s,cube=cube_in[iii],return_full=False,bg_off = bkg_off,verbose=verbose,medium=self.sim_med)
                    cube_seq.append(cube_out)
                    if verbose==True: 
                        count+=1
                        f.value = count
            else:
                cube_seq=[self.fp.get_fp(dit*u.s,cube=cube_in,return_full=False,bg_off = bkg_off,verbose=verbose,medium=self.sim_med)]
                if verbose==True: 
                    count+=1
                    f.value = count
                    
        if verbose==True:
            # Finish the progress bar
            f.bar_style = 'success'
            f.close()
            label.close()
        cube_seq=np.array(cube_seq)
        
        ###bin down cube
        if self.sim_med==False:
            self.lams = np.linspace(self.wav_min,self.wav_max,54)
            new_cube = np.zeros([len(cube_seq),54,108,108])
        if self.sim_med==True:
            self.lams = np.linspace(self.wav_min,self.wav_max,1900)
            new_cube = np.zeros([len(cube_seq),1900,len(cube_seq[0][0]),len(cube_seq[0][0][0])])


        print('binning down oversampled cube')
        for k in range(len(cube_seq)):
            for i in range(len(cube_seq[0][0])):
                for j in range(len(cube_seq[0][0][0])):
                    old_spec_wavs = self.lens.Prism.ll.value
                    new_spec_wavs = self.lams
                    spec_fluxes = cube_seq[k,:,i,j]
                    new_cube[k,:,i,j] = spectbin(new_spec_wavs, old_spec_wavs, spec_fluxes)

        if shot_off==False:
            new_cube = np.random.poisson(new_cube)
        
        
        if return_oversampled == True: return new_cube, self.lams, cube_seq
        else: return new_cube, self.lams



    def rebin_scene_bylam(self,wav,scene_in):
        new_cube = np.zeros([len(scene_in),len(self.lens.Prism.ll),108,108])
        for k in range(len(scene_in)):
            for i in range(108):
                for j in range(108):
                    new_spec_wavs = self.lens.Prism.ll.value
                    spec_fluxes = scene_in[k,:,i,j]
                    new_cube[k,:,i,j] = spectres(new_spec_wavs, wav, spec_fluxes)
        return new_cube
            