'''
demo code, simulate a Lyot coronagraphic image, both diffraction-limited and with aberration
'''

import matplotlib.pyplot as plt
from scipy.ndimage import shift
import numpy as np

p3i=lambda i: int(round(i)) #python2 to 3: change indicies that are floats to integers

wav0=1.65e-6 #assumed wav0 for sine amplitude input in meters

imagepix=1024
beam_ratio=5 #pixels/resel
pupilpix=p3i(round(imagepix/beam_ratio))

grid=np.mgrid[0:imagepix,0:imagepix]
xcen,ycen=imagepix/2,imagepix/2
xy_dh=np.sqrt((grid[1]-imagepix/2.)**2.+(grid[0]-imagepix/2.)**2.)

aperture=np.zeros((imagepix,imagepix))
aperture[np.where(xy_dh<pupilpix/2)]=1. #unobscured aperture
indpup=np.where(aperture==1.)

fpm=np.ones(aperture.shape)
iwa=3.
fpm[np.where(xy_dh/beam_ratio<iwa)]=0.

lyot_stop=np.zeros(aperture.shape)
undersize=1-1/(iwa*2)
lyot_stop[np.where(xy_dh<pupilpix/2.*undersize)]=1. #slightly undersized pupil

pupil_to_image = lambda im:	np.fft.fft2(im,norm='ortho')
image_to_pupil = lambda im: np.fft.ifft2(im,norm='ortho')
intensity = lambda ef: np.abs(ef)**2

def propagate(phase_in):
	'''
	generate coronagraphic image
	'''
	pupil_wavefront=aperture*np.exp(1j*(phase_in)) #initial
	norm=np.max(intensity(np.fft.fftshift(pupil_to_image(pupil_wavefront*lyot_stop)))) #contrast normalization
	fpm_wavefront_ini=np.fft.fftshift(pupil_to_image(pupil_wavefront)) #ft to image plane
	fpm_wavefront=fpm_wavefront_ini*fpm
	lyot_pupil_wavefront=image_to_pupil(fpm_wavefront) #ift to pupil plane
	masked_lyot_pupil_wavefront = lyot_pupil_wavefront*lyot_stop #add lyot stop
	imnorm=intensity(pupil_to_image(masked_lyot_pupil_wavefront))
	im=imnorm/norm
	return im

#diffraction limited coronagraphic image
im_dl=propagate(np.zeros(aperture.shape))

def xy_plane(dim):
	'''
	define xy plane to use for future functions
	'''
	grid=np.mgrid[0:dim,0:dim]
	xy=np.sqrt((grid[0]-dim/2.)**2.+(grid[1]-dim/2.)**2.)
	return xy
def complex_amplitude(mag,phase):
	'''
	complex amplitude in terms of magnitude and phase
	'''
	return mag*np.cos(phase)+1j*mag*np.sin(phase)
def antialias(phin,imagepix,beam_ratio):
	'''
	anti-alias via a butterworth filter; assuming 32 actuators across the pupil
	'''
	xy=xy_plane(imagepix)
	buttf = lambda rgrid,eps,r0,n: 1./np.sqrt(1+eps**2.*(xy/r0)**n) #butterworth filter
	phinput=phin-np.min(phin)
	phfilt=np.abs(pupil_to_image(np.fft.fftshift(image_to_pupil(phinput))*(buttf(xy,1,32/2.*beam_ratio*0.99,100)))) #assuming 32 actuators
	phout=phfilt-np.mean(phfilt)
	return phout

def make_noise_pl(wavefronterror,imagepix,pupilpix,wavelength,pl):
	'''
	make noise with a user input power law:

	(1) take white noise in image plane
	(2) IFFT to pupil plane
	(3) weight complex pupil plane (spatial frequency) by -1 power law (power~amplitude**2~x**(-2), so amplitude goes as (x**(-2))**(1/2)=x(-1)
	(4) FFT back to image plane

	wavefronterror = rms WFE (nm), normalized within the AO control region

	'''

	white_noise=np.random.random((imagepix,imagepix))*2.*np.pi #phase
	xy=xy_plane(imagepix)
	amplitude=(xy+1)**(pl/2.) #amplitude central value in xy grid is one, so max val is one in power law, everything else lower

	amplitude[p3i(imagepix/2),p3i(imagepix/2)]=0. #remove piston

	#remove alaising effects by cutting off power law just before the edge of the image
	amplitude[np.where(xy>imagepix/2.-1)]=0.

	amp=shift(amplitude,(-imagepix/2.,-imagepix/2.),mode='wrap')
	image_wavefront=complex_amplitude(amp,white_noise)
	noise_wavefront=np.real(np.fft.fft2(image_wavefront))

	beam_ratio=p3i(imagepix/pupilpix)

	norm_factor=(wavefronterror/wavelength*2.*np.pi)/np.std(antialias(noise_wavefront,imagepix,beam_ratio)[np.where(xy<pupilpix/2.)]) #normalization factor for phase error over the pupil of modes within the DM control region
	phase_out_ini=noise_wavefront*norm_factor

	phase_out=phase_out_ini#remove_tt(phase_out_ini,imagepix,pupilpix) #tip tilt removed phase screen

	return phase_out

ao_res=make_noise_pl(200e-9,imagepix,pupilpix,wav0,-2) #ao residual phase screen
static=make_noise_pl(100e-9,imagepix,pupilpix,wav0,-2) #static aberrations

im_abber=propagate(ao_res+static)


f = plt.figure()
plt.subplots_adjust(wspace=0)
ax=f.add_subplot(121)
plt.imshow(im_dl[300:700,300:700])
ax.set_yticks([])
ax.set_xticks([])
ax=f.add_subplot(122)
plt.imshow(im_abber[300:700,300:700])
ax.set_yticks([])
ax.set_xticks([])
plt.savefig('ben_example.png')
#plt.show()
