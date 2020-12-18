from astropy import units as u
import numpy as np
import astropy.io.fits as pyfits

def phoenix_star(T_s,logg,zz,rstar,dstar):
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

    specstar = pyfits.getdata('PHOENIX_HiRes/lte'+str(T_s)+'-'+str(logg)+'-'+str(zz)+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits') #flux units are erg/s/cm^2/cm
    wav = pyfits.getdata('PHOENIX_HiRes/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits') #units are angstrom
    wav = wav / 1.0e4
    rstar = rstar*u.R_sun.to(u.cm)
    dstar = dstar*u.pc.to(u.cm)
    fluxs = specstar * (rstar / dstar)**2 * np.pi * 1.0e-4 ##to convert from /cm to /um
    return wav,fluxs
