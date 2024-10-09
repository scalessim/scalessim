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
    if T_s < 10000: T_s = '0'+str(T_s)

    #specstar = pyfits.getdata('PHOENIX_HiRes/lte'+str(T_s)+'-'+str(logg)+'-'+str(zz)+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits') #flux units are erg/s/cm^2/cm
    specstar = pyfits.getdata(phoenixdir+'lte'+str(T_s)+'-'+str(logg)+'-'+str(zz)+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits') #flux units are erg/s/cm^2/cm
    wav = pyfits.getdata(phoenixdir+'/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits') #units are angstrom
    wav = wav / 1.0e4 ##now units are um
    rstar = rstar*u.R_sun.to(u.cm) 
    dstar = dstar*u.pc.to(u.cm)
    fluxs = specstar * (rstar / dstar)**2 * 1.0e-4 ##to convert from /cm to /um
    return wav,fluxs

def sonora_planet(T_p=300,sg=100,rp=1.0,d=10.0):
    rjup_cm = 6.9911e9
    rplan_cm = rp*rjup_cm
    pc_cm = 3.086e18
    a = np.loadtxt('../SCALES/Science_Reqs/cases/exo_disc/sonora_2018/spectra/sp_t'+str(T_p)+'g'+str(sg)+'nc_m0.0.gz',skiprows=2) 
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
    return wav,flam

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

            


