from astropy import units as u
import numpy as np
import astropy.io.fits as pyfits
from .io import *

def phoenix_star(T_s = 3800,logg = 4.5,zz = 0.0,rstar = 1.0,dstar = 20,phoenixdir='data/PHOENIX_HiRes/'):
    """
    inputs:
        T_s - effectve temperature in K
        logg - log surface gravity in cgs
        zz - metallicity
        rstar - stellar radius in R_sun
        dstar - distance to the system in pc

    returns:
        wav - list of wavelengths in microns
        fluxs - specific intensity in erg/s/cm^2/micron
    """
    logg = '%.2f' % logg
    if T_s < 10000: T_s = '0'+str(int(T_s))

    #specstar = pyfits.getdata('PHOENIX_HiRes/lte'+str(T_s)+'-'+str(logg)+'-'+str(zz)+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits') #flux units are erg/s/cm^2/cm
    specstar = pyfits.getdata(phoenixdir+'lte'+str(T_s)+'-'+str(logg)+'-'+str(zz)+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits') #flux units are erg/s/cm^2/cm
    wav = pyfits.getdata(phoenixdir+'/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits') #units are angstrom
    wav = wav / 1.0e4 ##now units are um
    rstar = rstar*u.R_sun.to(u.cm) 
    dstar = dstar*u.pc.to(u.cm)
    fluxs = specstar * (rstar / dstar)**2 * 1.0e-4 ##to convert from /cm to /um
    targ = Target(wav,fluxs)
    return targ

def sonora_planet(T_p=300,sg=100,rp=1.0,d=10.0,sonoradir = 'data/sonora_2018/'):
    rjup_cm = 6.9911e9
    rplan_cm = rp*rjup_cm
    pc_cm = 3.086e18
    a = np.loadtxt(sonoradir+'spectra/sp_t'+str(int(T_p))+'g'+str(sg)+'nc_m0.0.gz',skiprows=2) 
    #microns, fnu in cgs through surface (erg / s /cm^2 / Hz), lam in microns
    a = a[::-1]
    a_lo = a[np.where(a[:,0] > 1.0)]
    a_sub = a_lo[np.where(a_lo[:,0] < 5.5)]
    lplan_um = a_sub[:,0]
    
    a_dist = a_sub.copy()
    a_dist[:,1] = (rplan_cm / (d*pc_cm))**2 * a_dist[:,1] * np.pi
    
    ###convert fnu to flambda
    C = 2.998e10 ###cgs speed of light
    flam_tmp = a_dist[:,1] * C / (a_dist[:,0]*1.0e-4)**2 ###now in erg / s / cm^2 / cm
    flam = flam_tmp * 1.0e-4 ###now in erg / s / cm^2 / um
    wav = a_dist[:,0]
    targ = Target(wav,flam)
    return targ

def star_and_bkg(T_s=3800,logg=4.5,zz=0.0,rstar=1.0,dstar=20,Lmag=None,Mmag=None):
    wav,I_lam = phoenix_star(T_s=T_s,logg=logg,zz=zz,rstar=rstar,dstar=dstar)

    Lflux = I_lam[np.where(np.abs(wav-3.8)==np.min(np.abs(wav-3.8)))]
    Mflux = I_lam[np.where(np.abs(wav-5.0)==np.min(np.abs(wav-5.0)))]
    Hflux = I_lam[np.where(np.abs(wav-1.6)==np.min(np.abs(wav-1.6)))]

    Lmag_t = -2.5*np.log10(Lflux/(2.2272e-7))
    Mmag_t = -2.5*np.log10(Mflux/(1.222e-7))
    Hmag_t = -2.5*np.log10(Hflux/(1.22e-6))

    ####set this up as a Target
    if Lmag!=None:
        targ = Target(wav,I_lam*(10**(-(Lmag-Lmag_t)/2.5)))
    else: targ = Target(wav,I_lam)

    if Mmag!=None:
        targ = Target(wav,I_lam*(10**(-(Mmag-Mmag_t)/2.5)))
    targ_bg = Target(wav,np.zeros(wav.shape))
    return targ, targ_bg


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    angle1 = np.deg2rad(angle)
    qx = ox + np.cos(angle1) * (px - ox) - np.sin(angle1) * (py - oy)
    qy = oy + np.sin(angle1) * (px - ox) + np.cos(angle1) * (py - oy)
    return qy, qx


def planet_ADI_scene_lowres(psfs,
                     T_s=3800,logg_s=4.5,zz_s=0.0,r_s=1.0,Lmag=None,Mmag=None,
                     T_p=1000,sg_p=100,r_p=1.0,
                     d=10.0,
                     PAlist=np.linspace(-45,45,90),p_sep=350.0, p_PA=45.0,
                     lamlist=np.linspace(1.9,5.3,341),
                     psfs_coron=None,vortex=False):
    
    star = phoenix_star(T_s=T_s,logg=logg_s,zz=zz_s,rstar=r_s,dstar=d)
    planet = sonora_planet(T_p=T_p,sg=sg_p,rp=r_p,d=d)

    star_new = spectres(lamlist, star.x.value, star.y.value)
    planet_new = spectres(lamlist, planet.x.value, planet.y.value)

    scene = np.zeros([len(PAlist),len(star_new), 108, 108])
    seps=np.array([p_sep/20.0])
    position_angles=np.deg2rad([p_PA])
    posns = np.array([54+seps*np.cos(-position_angles), 54+seps*np.sin(-position_angles)]).T
    coords = rotate((54,54),posns[0],PAlist)

    #plt.scatter(coords[0],coords[1])
    #plt.show()

    scene_conv = np.zeros(scene.shape)

    if vortex==False:
        for i in range(len(scene)):
            scene[i,:,int(coords[1][i]),int(coords[0][i])] = planet_new
            scene[i,:,54,54] = star_new
            for j in range(len(scene[i])):
                FTPSF = np.fft.fft2(np.fft.fftshift(psfs[i,j]))
                FTSCENE = np.fft.fft2(np.fft.fftshift(scene[i,j]))
                FTCONV = FTPSF*FTSCENE
                #FTPSF = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psfs[i,j])))
                #FTSCENE = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(scene[i,j])))
                convim = np.fft.ifftshift(np.fft.ifft2(FTCONV))
                scene_conv[i][j] = np.real(convim)
        scene_conv = np.array(scene_conv)*u.erg/u.cm/u.cm/u.s/u.um
        return scene, scene_conv
    else:
        scene_conv_s = np.zeros(scene.shape)
        for i in range(len(scene)):
            sc_star = scene[i].copy()
            sc_star[:,54,54] = star_new
            sc_planet = scene[i].copy()
            sc_planet[:,int(coords[1][i]),int(coords[0][i])] = planet_new
            for j in range(len(scene[i])):
                FTPSF = np.fft.fft2(np.fft.fftshift(psfs[i,j]))
                FTPSF_C = np.fft.fft2(np.fft.fftshift(psfs_coron[i,j]))
                FTSCENE_S = np.fft.fft2(np.fft.fftshift(sc_star[j]))
                FTSCENE_P = np.fft.fft2(np.fft.fftshift(sc_planet[j]))
                FTCONV_S = FTSCENE_S * FTPSF_C
                FTCONV_P = FTSCENE_P * FTPSF
                convim_s = np.fft.ifftshift(np.fft.ifft2(FTCONV_S))
                convim_p = np.fft.ifftshift(np.fft.ifft2(FTCONV_P))
                scene_conv[i][j] = np.real(convim_s + convim_p)

                convim_sn = np.fft.ifftshift(np.fft.ifft2(FTPSF*FTSCENE_S))
                scene_conv_s[i][j] = np.real(convim_sn)
            scene[i] = sc_star+sc_planet

        scene_conv = np.array(scene_conv)*u.erg/u.cm/u.cm/u.s/u.um
        scene_conv_s = np.array(scene_conv_s)*u.erg/u.cm/u.cm/u.s/u.um
        return scene, scene_conv, scene_conv_s
    
            


