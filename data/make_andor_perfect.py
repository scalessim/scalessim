import numpy as np

lminlist = [1.95,2.0,3.1,2.0,4.5,2.9]
lmaxlist = [2.45,4.0,3.5,5.2,5.2,4.15]

for i in range(len(lminlist)):
    lmin = lminlist[i]
    lmax = lmaxlist[i]
    a = np.loadtxt('andor_'+str(lmin)+'_'+str(lmax)+'.txt')
    b = a.copy()
    b[:,1][np.where(b[:,0]<=lmin)] = 0
    b[:,1][np.where(b[:,0]>=lmax)] = 0
    np.savetxt('andor_'+str(lmin)+'_'+str(lmax)+'_perfect.txt',np.array(b))
