import numpy as np

def DFT_matrix(N, M):
    """Calculate dft matrix
    N : number of pixels
    M : magnification factor"""

    i, j = np.meshgrid(np.arange(N)-N/2, np.arange(N)-N/2)
    omega = np.exp( - 2 * np.pi * 1.J / N / M )
    W = np.power( omega, i * j ) / np.sqrt(N)
    return W
