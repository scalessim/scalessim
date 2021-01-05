#####make prism smaller to make things run faster!
#####could do ~110 instead of 341


import numpy as np

pd = np.loadtxt('L_prism.txt')
lams = pd[:,0]


lams2 = lams[::5]
pd2 = pd[:,1]
ps2 = pd2[::5]


ofile = 'L_prism_coarse.txt'
towrite = np.array([[lams2[x],ps2[x]] for x in range(len(lams2))])
print(towrite.shape)
np.savetxt(ofile,towrite)
