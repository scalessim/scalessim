import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits


class Pupil(object):
    def __init__(self):
        self.pixsize = 0
        self.d = 10
        self.rate = 2.5e-3
        self.pupil = None
        self.area = None

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, _rate):
        self._rate = _rate
        self.pixsize = int(self.d * 1.0 /self._rate)

    def make_pupil(self):
        print('make_pupil not works for Pupil object')

    def load_from(self, file):
        img, header = fits.getdata(file,header=True)
        self.pixsize = img.shape[0]
        self.rate = header['rate']
        self.pupil = img
        self.d = header['diameter']
        self.type = header['type']

    def save_to(self, file):
        head = fits.Header()
        head['diameter'] = self.d
        head['rate'] = self.rate
        head['type'] = self.type
        fits.writeto(file, self.pupil, header = head, overwrite = True)

    def reduce_rate(self,rate):
        if rate > 1:
            img = self.pupil
            sz = np.array(img.shape)
            fimg = np.fft.fftshift(np.fft.fft2(img))
            new_sz = (sz/rate).astype(int)
            ori = ((sz-new_sz)/2).astype(int)

            sub_fimg = fimg[ori[0]:ori[0]+new_sz[0],ori[1]:ori[1]+new_sz[1]]
            sub_img = np.fft.ifft2(sub_fimg)
            self.rate *= float(sz[0]) / new_sz[0]
            self.pupil = abs(sub_img) / self.rate**2
        elif rate < 1:
            raise Exception('Cant imcrease rate!')

    def show(self, ax = plt):
        ax.imshow(self.pupil.T, origin = 'lowerdef ')
