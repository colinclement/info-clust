from __future__ import print_function
import numpy as np
from scipy.misc import logsumexp
#from _cuClust import optimizeP
import sgc.bialekClustering as bcl
from datetime import datetime

cij = np.load('cij-beta_1p42-dis_992523170.npz')['cij']
#cij = np.load("cij-beta_5-dis-1332369787.npz")['cij']
sij = np.abs(cij)
N_c, N = 64, cij.shape[0] 
P = np.random.rand(N_c, N)
P /= P.sum(0)
P0 = P.copy()
T = 0.015
eps = 1E-7
maxiter = N

start = datetime.now()
bialek = bcl.BialekClustering(sij, N_c, T, P = P.copy(), iprint = 2,
                              maxiter = maxiter, target = 'cpu')#, eps = eps, maciter = maxiter)
print("Numpy clustering time: {}".format(datetime.now()-start))

start = datetime.now()
#niters = optimizeP(sij, P, T, eps, maxiter, 2)
cu_bialek = bcl.BialekClustering(sij, N_c, T, P = P.copy(), iprint = 2,
                                 maxiter = maxiter, target = 'gpu')#, eps = eps, maciter = maxiter)
print("Cuda clustering time: {}".format(datetime.now()-start))

