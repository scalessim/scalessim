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
