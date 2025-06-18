import os
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.modeling.models import BlackBody
import glob
from scipy.io import readsav
from scipy import interpolate

class DataFile:
    def __init__(self):
        self.filename = ''
        self.default_name = ''

    def get_data(self, xunits=u.micron, yunits=u.erg/u.s/u.cm**2/u.micron):
        try:
            self.x, self.y = np.loadtxt('./data/{}'.format(self.filename), unpack=True)
        except:
            print('Could not find {}, using {}'.format(self.filename, self.default_name))
            self.x, self.y = np.loadtxt('./data/{}'.format(self.default_name), unpack=True)
            self.filename = self.default_name
        self.x = self.x * xunits
        self.y = self.y * yunits

    def to(self, to_xunits=u.micron, to_yunits=u.photon/u.s/u.cm**2/u.micron, equivalencies=None):
        equivalencies = equivalencies or u.spectral_density(self.x)
        self.x = self.x.to(to_xunits, equivalencies=u.spectral())
        self.y = self.y.to(to_yunits, equivalencies=equivalencies)

    def interp(self, new_wavelengths):
        wav = new_wavelengths.to(self.x.unit)
        return np.interp(wav.value, self.x.value, self.y.value)*self.y.unit

    def resample(self, new_wavelengths):
        wav = new_wavelengths.to(self.x.unit)
        return spectres(wav.value, self.x.value, self.y.value)*self.y.unit

class Vega(DataFile):
    def __init__(self):
        self.filename = 'vega_kurucz.txt'
        self.get_data()
        self.to()

class Target(DataFile):
    def __init__(self, x,y):
            self.x = x*u.micron
            self.y = y*u.erg / u.cm**2 / u.micron / u.s

class SkyBG(DataFile):
    def __init__(self, vapor, airmass, flag='mk'):
        if flag == 'mk':
            va = min([10, 16, 30, 50], key=lambda x:abs(x - vapor*10))
            am = min([10, 15, 20], key=lambda x:abs(x - airmass*10))
            self.filename = "skybg/%s_skybg_zm_%d_%d_ph.dat"%(flag, va, am)
            self.default_name = "skybg/mk_skybg_zm_10_10_ph.dat"
            self.get_data(xunits=u.nm, yunits=u.photon/u.s/ u.nm / u.m**2) #the /arcsec**2 screws up the unit conversion
        elif flag == 'cp':
            va = min([23, 43, 76, 100], key=lambda x:abs(x - vapor*10))
            am = min([10, 15, 20], key=lambda x:abs(x - airmass*10))
            self.filename = "skybg/%s_skybg_zm_%d_%d_ph.dat"%(flag, va, am)
            self.default_name = "skybg/cp_skybg_zm_23_10_ph.dat"
            self.get_data(xunits=u.nm, yunits=u.photon/u.s / u.nm / u.m**2) #the /arcsec**2 screws up the unit conversion
        else:
            raise ValueError('Site model {} does not exist. Choose mk or cp'.format(flat))
        self.to()
        self.y /= u.arcsec**2  #added back /arcsec**2


class SkyTrans(DataFile):
    def __init__(self, vapor, airmass, flag='mk'):
        if flag == 'mk':
            va = min([10, 16, 30, 50], key=lambda x:abs(x - vapor*10))
            am = min([10, 15, 20], key=lambda x:abs(x - airmass*10))
            self.filename = "skytrans/%strans_zm_%d_%d.dat"%(flag, va, am)
            self.default_name = "skytrans/mktrans_zm_10_10.dat"
            self.get_data(xunits=u.micron, yunits=u.dimensionless_unscaled)
        elif flag == 'cp':
            va = min([23, 43, 76, 100], key=lambda x:abs(x - vapor*10))
            am = min([10, 15, 20], key=lambda x:abs(x - airmass*10))
            self.filename = "skytrans/%strans_zm_%d_%d.dat"%(flag, va, am)
            self.default_name = "skytrans/cptrans_zm_23_10.dat"
            self.get_data(xunits=u.micron, yunits=u.dimensionless_unscaled)
        else:
            raise ValueError('Site model {} does not exist. Choose mk or cp'.format(flat))
        self.to(to_yunits=u.dimensionless_unscaled, equivalencies=None)

class AtmoDispersion(DataFile):
    def __init__(self, RH, T, P):
        rh = min(np.arange(10,100,10), key=lambda x:abs(x - RH))
        t = min(np.arange(-15,25,5), key=lambda x:abs(x - T))
        p = min(np.arange(550, 675, 25), key=lambda x:abs(x - P))

        self.filename = "atmospheric_dispersion_dat/airmass2.0_%d_%d_%d.dat"%(rh, t, p)
        self.default_name = "atmospheric_dispersion_dat/airmass2.0_10_-10_700.dat"
        self.get_data(xunits=u.micron, yunits=.001*u.arcsec)
        self._airmass = 2.0

    @property
    def airmass(self):
        return self._airmass

    @airmass.setter
    def airmass(self, _airmass):
        self.y *= np.tan(np.arccos(1/_airmass)) / np.tan(np.arccos(1./self._airmass))
        self._airmass = _airmass

    def get(self, wavelengths):
        re = super().resample(wavelengths)
        re -= re.mean()
        return re

class InstTransEm(DataFile):
    def __init__(self, tel_AO_ems=[.4], temps=[285*u.K], sc_trans = [0.4]):
        self.eps = tel_AO_ems
        self.temps = temps
        self.itrans = sc_trans

    def load(self, filename):
        self.filename = filename
        self.get_data(xunits=u.dimensionless_unscaled, yunits=u.K)
        self.eps = self.x
        self.temps = self.y

    def BB(self, temp, wavelengths):
        #flux = blackbody_lambda(wavelengths, temp)*u.sr
        bb = BlackBody(temperature=temp,scale=1.0*u.erg / (u.cm ** 2 * u.s * u.AA * u.sr))
        flux = bb(wavelengths)
        out = flux.to(u.photon/u.s/u.cm**2/u.micron/(u.arcsec**2), equivalencies=u.spectral_density(wavelengths))
        return out



    def get_trans(self, wavelengths):
        """
        returns the transmissions of the telescope + instrument (separately)

        tel_AO_trans = 1 - tel_AO_ems is your telescope transmission
        scales_trans = sc_trans
        """

        trans = 1
        for eps in self.eps:
            trans *= (1-eps)  #tel/AO transmission
        tel_AO_trans = np.array([trans]*len(wavelengths)) * u.dimensionless_unscaled
        itrans = 1
        for tt in self.itrans:
            itrans *= tt
        scales_trans = np.array([itrans]*len(wavelengths)) * u.dimensionless_unscaled
        return tel_AO_trans, scales_trans

    def get_em(self, wavelengths):
        """
        returns telescope + AO emission using the tel/AO emissivity (eps)
        and a blackbody with a specified temperature

        """
        key = 0
        for eps, temp in zip(self.eps, self.temps):
            Oi = eps * self.BB(temp, wavelengths)

            if key:
                Oall = (Oall * eps + Oi)
            else:
                key = 1
                Oall = Oi
        return Oall

class QE(DataFile):
    """
    returns an array of quantum efficiencies with the same shape as the input
    wavelengths. if a single number is given for qe, then an array containing
    uniform qes is returned.
    """
    def __init__(self, qe=.7):
        self.qe = qe

    def get_qe(self, wavelengths):
        return np.ones_like(wavelengths.value)*self.qe*u.electron / u.photon

class Prism(DataFile):
    def __init__(self, lmin, lmax):
        self.filename = 'dispersion_curves/'+str(lmin)+'_'+str(lmax)+'_prism.txt'
        #print(self.filename)
        if os.path.isfile('./data/{}'.format(self.filename))==True:
            #####need to add ys to OG prism files
            self.ll, self.x, self.y = np.loadtxt('./data/{}'.format(self.filename), unpack=True)
            ###units of dispersion curve x and y are mm!!!
            lams_des = lams_binned=np.linspace(1.9,5.3,341)
            xinterp = interpolate.interp1d(self.ll,self.x,kind='cubic')
            yinterp = interpolate.interp1d(self.ll,self.y,kind='cubic')
            x2 = xinterp(lams_des)
            y2 = yinterp(lams_des)
            self.ll = lams_des*u.micron
            self.x = x2*1000.0 ##converting to microns
            self.y = y2*1000.0 ##converting to microns
        else:
            print('no prism data!')
            stop

    #def scale_to_length(self, l):
    #    self.y -= self.y.min()
    #    self.y /= self.y.max()
    #    self.y *= l

    def get_dlam(self):
        return np.gradient(self.ll.value) * self.ll.unit

class Grating(DataFile):
    def __init__(self, lmin, lmax):
        self.filename = 'dispersion_curves/'+str(lmin)+'_'+str(lmax)+'_grat.txt'
        #print(self.filename)
        if os.path.isfile('./data/{}'.format(self.filename))==True:
            #####need to add ys to OG prism files
            self.ll, self.x, self.y = np.loadtxt('./data/{}'.format(self.filename), unpack=True)
            ###units of dispersion curve x and y are mm!!!
            lams_des = lams_binned=np.linspace(1.9,5.3,34001)
            xinterp = interpolate.interp1d(self.ll,self.x,kind='cubic')
            yinterp = interpolate.interp1d(self.ll,self.y,kind='cubic')
            x2 = xinterp(lams_des)
            y2 = yinterp(lams_des)
            self.ll = lams_des*u.micron
            self.x = x2*1000.0 ##Grism file is in microns
            self.y = y2*1000.0 ##Grism file is in microns
        else:
            print('no prism data!')
            stop

    #def scale_to_length(self, l):
    #    self.y -= self.y.min()
    #    self.y /= self.y.max()
    #    self.y *= l

    def get_dlam(self):
        return np.gradient(self.ll.value) * self.ll.unit

class Filter(DataFile):
    def __init__(self, fkw = 'filter_perfect',lmin = 2.0, lmax = 5.2, od = -100): #filename='L_filter.txt'):
        filename = 'ifs_filters/'+fkw+'_'+str(lmin)+'_'+str(lmax)+'.txt'
        self.filename = filename
        #if filter_name == 'L':
        #    self.filename = 'L_filter.txt'
        #else:
        #    raise ValueError('No filter data exists for filter {}'.format(filter_name))
        self.get_data(yunits=u.dimensionless_unscaled)

class ImagerFilter(DataFile):
    def __init__(self, filename='nirc2_Lp.txt'):
        self.filename = 'imager_filters/'+filename
        self.get_data(yunits=u.dimensionless_unscaled)


def read_ini(section):
    arg = {}
    for key in section:
        value = section[key]
        if '.' in value:
            arg[key] = float(value)
        else:
            arg[key] = int(value)
    return arg



def spectres(new_spec_wavs, old_spec_wavs, spec_fluxes, spec_errs=None):

    """
    Function for resampling spectra (and optionally associated uncertainties) onto a new wavelength basis.
    Parameters
    ----------
    new_spec_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the spectrum or spectra.
    old_spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the spectrum or spectra.
    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in old_spec_wavs, last dimension must correspond to the shape of old_spec_wavs.
        Extra dimensions before this may be used to include multiple spectra.
    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties associated with each spectral flux value.

    Returns
    -------
    resampled_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same length as new_spec_wavs, other dimensions are the same as spec_fluxes
    resampled_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in resampled_fluxes. Only returned if spec_errs was specified.
    """

    # Generate arrays of left hand side positions and widths for the old and new bins
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_widths = np.zeros(old_spec_wavs.shape[0])
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_lhs[0] = old_spec_wavs[0] - (old_spec_wavs[1] - old_spec_wavs[0])/2
    spec_widths[-1] = (old_spec_wavs[-1] - old_spec_wavs[-2])
    spec_lhs[1:] = (old_spec_wavs[1:] + old_spec_wavs[:-1])/2
    spec_widths[:-1] = spec_lhs[1:] - spec_lhs[:-1]

    filter_lhs = np.zeros(new_spec_wavs.shape[0]+1)
    filter_widths = np.zeros(new_spec_wavs.shape[0])
    filter_lhs[0] = new_spec_wavs[0] - (new_spec_wavs[1] - new_spec_wavs[0])/2
    filter_widths[-1] = (new_spec_wavs[-1] - new_spec_wavs[-2])
    filter_lhs[-1] = new_spec_wavs[-1] + (new_spec_wavs[-1] - new_spec_wavs[-2])/2
    filter_lhs[1:-1] = (new_spec_wavs[1:] + new_spec_wavs[:-1])/2
    filter_widths[:-1] = filter_lhs[1:-1] - filter_lhs[:-2]

    # Check that the range of wavelengths to be resampled_fluxes onto falls within the initial sampling region
    if filter_lhs[0] < spec_lhs[0] or filter_lhs[-1] > spec_lhs[-1]:
        raise ValueError("spectres: The new wavelengths specified must fall within the range of the old wavelength values.")

    #Generate output arrays to be populated
    resampled_fluxes = np.zeros(spec_fluxes[...,0].shape + new_spec_wavs.shape)

    if spec_errs is not None:
        if spec_errs.shape != spec_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape as spec_fluxes.")
        else:
            resampled_fluxes_errs = np.copy(resampled_fluxes)

    start = 0
    stop = 0

    # Calculate the new spectral flux and uncertainty values, loop over the new bins
    for j in range(new_spec_wavs.shape[0]):

        # Find the first old bin which is partially covered by the new bin
        while spec_lhs[start+1] <= filter_lhs[j]:
            start += 1

        # Find the last old bin which is partially covered by the new bin
        while spec_lhs[stop+1] < filter_lhs[j+1]:
            stop += 1

        # If the new bin falls entirely within one old bin the are the same the new flux and new error are the same as for that bin
        if stop == start:

            resampled_fluxes[...,j] = spec_fluxes[...,start]
            if spec_errs is not None:
                resampled_fluxes_errs[...,j] = spec_errs[...,start]

        # Otherwise multiply the first and last old bin widths by P_ij, all the ones in between have P_ij = 1
        else:

            start_factor = (spec_lhs[start+1] - filter_lhs[j])/(spec_lhs[start+1] - spec_lhs[start])
            end_factor = (filter_lhs[j+1] - spec_lhs[stop])/(spec_lhs[stop+1] - spec_lhs[stop])

            spec_widths[start] *= start_factor
            spec_widths[stop] *= end_factor

            # Populate the resampled_fluxes spectrum and uncertainty arrays
            resampled_fluxes[...,j] = np.sum(spec_widths[start:stop+1]*spec_fluxes[...,start:stop+1], axis=-1)/np.sum(spec_widths[start:stop+1])

            if spec_errs is not None:
                resampled_fluxes_errs[...,j] = np.sqrt(np.sum((spec_widths[start:stop+1]*spec_errs[...,start:stop+1])**2, axis=-1))/np.sum(spec_widths[start:stop+1])

            # Put back the old bin widths to their initial values for later use
            spec_widths[start] /= start_factor
            spec_widths[stop] /= end_factor


    # If errors were supplied return the resampled_fluxes spectrum and error arrays
    if spec_errs is not None:
        return resampled_fluxes, resampled_fluxes_errs

    # Otherwise just return the resampled_fluxes spectrum array
    else:
        return resampled_fluxes
    
    
def spectbin(new_spec_wavs, old_spec_wavs, spec_fluxes, spec_errs=None):

    """
    Function for resampling spectra (and optionally associated uncertainties) onto a new wavelength basis.
    Parameters
    ----------
    new_spec_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the spectrum or spectra.
    old_spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the spectrum or spectra.
    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in old_spec_wavs, last dimension must correspond to the shape of old_spec_wavs.
        Extra dimensions before this may be used to include multiple spectra.
    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties associated with each spectral flux value.

    Returns
    -------
    resampled_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same length as new_spec_wavs, other dimensions are the same as spec_fluxes
    resampled_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in resampled_fluxes. Only returned if spec_errs was specified.
    """

    # Generate arrays of left hand side positions and widths for the old and new bins
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_widths = np.zeros(old_spec_wavs.shape[0])
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_lhs[0] = old_spec_wavs[0] - (old_spec_wavs[1] - old_spec_wavs[0])/2
    spec_widths[-1] = (old_spec_wavs[-1] - old_spec_wavs[-2])
    spec_lhs[1:] = (old_spec_wavs[1:] + old_spec_wavs[:-1])/2
    spec_widths[:-1] = spec_lhs[1:] - spec_lhs[:-1]

    filter_lhs = np.zeros(new_spec_wavs.shape[0]+1)
    filter_widths = np.zeros(new_spec_wavs.shape[0])
    filter_lhs[0] = new_spec_wavs[0] - (new_spec_wavs[1] - new_spec_wavs[0])/2
    filter_widths[-1] = (new_spec_wavs[-1] - new_spec_wavs[-2])
    filter_lhs[-1] = new_spec_wavs[-1] + (new_spec_wavs[-1] - new_spec_wavs[-2])/2
    filter_lhs[1:-1] = (new_spec_wavs[1:] + new_spec_wavs[:-1])/2
    filter_widths[:-1] = filter_lhs[1:-1] - filter_lhs[:-2]

    # Check that the range of wavelengths to be resampled_fluxes onto falls within the initial sampling region
    if filter_lhs[0] < spec_lhs[0] or filter_lhs[-1] > spec_lhs[-1]:
        raise ValueError("spectres: The new wavelengths specified must fall within the range of the old wavelength values.")

    #Generate output arrays to be populated
    resampled_fluxes = np.zeros(spec_fluxes[...,0].shape + new_spec_wavs.shape)

    if spec_errs is not None:
        if spec_errs.shape != spec_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape as spec_fluxes.")
        else:
            resampled_fluxes_errs = np.copy(resampled_fluxes)

    start = 0
    stop = 0

    # Calculate the new spectral flux and uncertainty values, loop over the new bins
    for j in range(new_spec_wavs.shape[0]):

        # Find the first old bin which is partially covered by the new bin
        while spec_lhs[start+1] <= filter_lhs[j]:
            start += 1

        # Find the last old bin which is partially covered by the new bin
        while spec_lhs[stop+1] < filter_lhs[j+1]:
            stop += 1

        # If the new bin falls entirely within one old bin the are the same the new flux and new error are the same as for that bin
        if stop == start:

            resampled_fluxes[...,j] = spec_fluxes[...,start]
            if spec_errs is not None:
                resampled_fluxes_errs[...,j] = spec_errs[...,start]

        # Otherwise multiply the first and last old bin widths by P_ij, all the ones in between have P_ij = 1
        else:

            start_factor = (spec_lhs[start+1] - filter_lhs[j])/(spec_lhs[start+1] - spec_lhs[start])
            end_factor = (filter_lhs[j+1] - spec_lhs[stop])/(spec_lhs[stop+1] - spec_lhs[stop])

            spec_widths[start] *= start_factor
            spec_widths[stop] *= end_factor
            
            spec_weights = np.ones(spec_widths.shape)
            spec_weights[start] = start_factor
            spec_weights[stop] = end_factor

            # Populate the resampled_fluxes spectrum and uncertainty arrays
            resampled_fluxes[...,j] = np.sum(spec_weights[start:stop+1]*spec_fluxes[...,start:stop+1], axis=-1)

            if spec_errs is not None:
                resampled_fluxes_errs[...,j] = np.sqrt(np.sum((spec_weights[start:stop+1]*spec_errs[...,start:stop+1])**2, axis=-1))

            # Put back the old bin widths to their initial values for later use
            spec_widths[start] /= start_factor
            spec_widths[stop] /= end_factor


    # If errors were supplied return the resampled_fluxes spectrum and error arrays
    if spec_errs is not None:
        return resampled_fluxes, resampled_fluxes_errs

    # Otherwise just return the resampled_fluxes spectrum array
    else:
        return resampled_fluxes






