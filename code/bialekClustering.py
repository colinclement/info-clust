"""
bialekClustering.py
    
    This is a script written to implement 'Information based clustering'
    by Noam Slonim, Gurinder Singh Atwal, Gasper Tkacik, and Bill Bialek.

    Only the special case of pair-wise similarity measures was implemented.

        reference: Noam Slonim, 18297-18302, doi: 10.1073/pnas.0507432102 
                   or
                   https://www.princeton.edu/~wbialek/our_papers/slonim+al_05b.pdf

    author: Colin Clement
    date: 06/13/14

    ALSO CONTAINS: Hard clustering from agglomerative clustering with Bialek's
    rate-distortion condition for optimizing choice of number of clusters.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rv_discrete
from itertools import chain
from scipy.misc import logsumexp

import sgc.clusterUtilities as clu
try:
    from sgc.cuClust._cuClust import optimizeP #CUDA
    hascudaclust = True
except ImportError:
    hascudaclust = False


#==============================
#   Bialek Clustering 
#==============================

class BialekClustering(object):
    """Implementation of Information-based Clustering"""
    def __init__(self, c_ij, N_c, T, eps = 1E-6,
                 maxiter = None, iprint = 0, **kwargs):
        """
        input:
            (required)
            c_ij : pairwise correlation matrix of shape (N, N)
            N_c : (int) number of clusters to consider
            T : (pos. float) parameter. Default is random in (0,1]
                {8: 0.03, 16, 0.012} seem effective for size L
            (optional)
            eps : (float) tolerance for convergence (default 1E-12)
            seed : (int) seed for random initial condition
            maxiter : (int) maximum number of iterations to allow
            iprint : (int)  print messages if iprint > 0

            **kwargs:
                P : initial probalities float(N_c, N)
                N_avg : Number of initial conditions to average over in
                        self.average_over_initial (default 10)
        
        """
        self.s_ij = self._cij_to_sij(c_ij)
        self.N, self.N_c = c_ij.shape[0], N_c
        self.rng = np.random.RandomState()
        self.T = T
        self.eps = eps
        self.maxiter = maxiter or 500 #self.N/2
        self.iprint = iprint

        #initialize
        self.seed = kwargs.get('seed', 920891485)
        self.rng.seed(self.seed)
        self.P = kwargs.get('P', self.rng.rand(self.N_c, self.N))
        self.N_avg = kwargs.get('N_avg', 4)
        self._target = kwargs.get('target', 'gpu')
        
        #self.P = self.average_over_initial(self.P)
        self.P = self.optimize()
        self.clust = self.assignClusterMaxProb(self.P)

    def _cij_to_sij(self, c_ij):
        """
        Symmetrizes and takes the absolute value of c_ij
        spin-spin correlation matrix.
        input: correlation matrix (N,N)
        output: symmetrized, diagonal-rounding removed
        """
        s_ij = np.abs((c_ij + c_ij.T)/2.)
        s_ij[np.diag_indices(s_ij.shape[0])] = 1.0
        return s_ij
    
    ##------------------
    # Objective function
    ##------------------

    def infoFreeEnergy(self, P_C_i, T):
        """
        input:  
            P_C_i : (N_c, N) P(cluster | index)
            T : (float) positive parameter
        output:
            float : information free energy
        """
        N = self.N
        NP_C = P_C_i.sum(axis=1)
        pci_by_npc = P_C_i/NP_C[:,None] 
        S = np.nansum(pci_by_npc*P_C_i.dot(self.s_ij))/N
        with np.errstate(divide = 'ignore', invalid = 'ignore'): 
            I = np.nansum(P_C_i*np.log(N*pci_by_npc))/N 
        return S - T * I

    ##============================
    # Iterative clustering process
    ##============================

    def optimize(self, T = None, P = None):
        """
        Performs Bialek information-based clustering. (see ref. above)
            T : parameter (in F = <S> - T I)
        output:
            P(C | i) : probability array of shape (N_c, N)
        """
        T = T or self.T
        P = P if P is not None else self.P #prob(cluster | index)
        P /= P.sum(axis=0) #normalize
        logP = np.log(P)
       
        if hascudaclust and self._target == 'gpu':
            if self.iprint: print "Optimizing on the GPU"
            niters = optimizeP(self.s_ij, P, T, 
                               self.eps, self.maxiter, self.iprint)
        else:
            if self.iprint:
                print "Optimizing on the CPU"
                initial_F = self.infoFreeEnergy(P, T)
                print '\nInitial F = {free}'.format(free=initial_F)
            
            finished = False
            m = 0
            s_ij, N = self.s_ij, self.N
            while not finished:
                Pold = P.copy()
                logP_C = logsumexp(logP, axis=1)
                NP_C = P.sum(axis=1)
                PS = P.dot(s_ij)
                logP = logP_C[:,None]+(PS/T)*(2-P/NP_C[:,None])/NP_C[:,None]
                logP -= logsumexp(logP, axis=0) #Normalize
                m += 1

                P = np.exp(logP)
                deltaP2 = np.sqrt(np.mean((P-Pold)**2))
                if self.iprint > 1:
                    print "\t{}: deltaP = {}".format(m, deltaP2)
                
                if deltaP2 < self.eps or m == self.maxiter:
                    if m == self.maxiter: print "Maximum iterations reached"
                    if self.iprint: 
                        final_F = self.infoFreeEnergy(P, T)
                        print 'Final F = {free}\n'.format(free=final_F)
                    finished = True
        
        return P
 
    def average_over_initial(self, P = None, T = None):
        """
        Averages over self.N_avg different (random) initial conditions,
        chooses solution which is the largest
        """
        if T == None:
            T = self.T
        Fmax = -np.inf #Always smaller than any real number
        Pmax = P
        for i in range(self.N_avg):
            if i == 1:
                P = P if P is not None else np.random.rand(self.N_c, self.N)
            else:
                P = np.random.rand(self.N_c, self.N)
            P = self.optimize(T = T, P = P)
            F = self.infoFreeEnergy(P, T)
            if F > Fmax:
                Fmax = F
                Pmax = P.copy()
        if T == self.T:
            self.P = Pmax
        return Pmax

    #==============================
    #   Rate Distortion
    #==============================
    
    def plot_rate_distortion(self, Tset):
        """
        Plots Information free energy versus parameter T in Tset.
        input:
            Tset : Array of T-values to consider
        output:
            plot as described
        """
        Pset = np.array([self.average_over_initial(T = T) for T in Tset])
        infoF = [self.infoFreeEnergy(P, T) for P, T in zip(Pset, Tset)]
        plt.figure()
        plt.plot(Tset, infoF)
        plt.xlabel('T parameter')
        plt.ylabel(r'$F=\langle S\rangle - T I$')
        plt.title('Rate distortion curve')
        return infoF

    #==============================
    #   Assigning clusters fro P(C|i)
    #==============================
    
    def assignClusterMaxProb(self, P):
        """
        Chooses clustering assignment based on maximum probability
        P(C|i) for each i.
        """
        labl = range(self.N_c)
        maxclust_assign = [np.argmax(i) for i in P.T]
        clust = [[] for n in labl]
        for nc in labl:
            for i, c in enumerate(maxclust_assign):
                if c == nc:
                    clust[nc] += [i]
        clust = [c for c in clust if c]
        #Split spatially disjoint clusters
        if np.sqrt(self.N)%1. == 0: #square two dimensional
            clust = self.splitClustersSpatially(clust) 
        return clust

    def splitClustersSpatially(self, clust):
        split = []
        L = int(np.sqrt(self.N))
        for c in clust:
            split += clu.spatialSplit(c, L)
        return split 

    def sampleClusterProb(self):
        """
        Samples directly from probability distribution P(C|i)
        """
        labl = range(self.N_c)
        assignment = []
        for pci in self.P.T:
            distrib = rv_discrete(values=(labl, pci))
            assignment += [distrib.rvs(size=1)]
        clust = [[] for n in labl]
        for nc in labl:
            for i, c in enumerate(assignment):
                if c == nc:
                    clust[nc] += [i]
        return [c for c in clust if c] #remove empties
   


