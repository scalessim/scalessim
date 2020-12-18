import numpy as np



#mag @ 2 micron = 28.12
#mag @ 5.2 micron = 73.125
####move these into a helper function!!
def magn(lam):
    dm = 73.125-28.12
    dl = 5.2-2.0
    mm = dm/dl
    mag = 28.12 + (lam-2.0)*mm
    return mag

