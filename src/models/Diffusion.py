'''
Created on Sep 1, 2016

@author: urishaham
'''
import numpy as np
from sklearn.neighbors import NearestNeighbors
#from scipy.sparse.linalg import svds

def Diffusion(K, nEigenVals = 12): # Laplace - Beltrami
    d = np.sum(K, axis = 0)
    d = np.sqrt(d)
    D = np.diag(d)
    D_inv = np.linalg.inv(D)
    A = np.dot(np.dot(D_inv,K), D_inv)
    Vals, Vecs = np.linalg.eig(A)
    # sort eigenvalues
    I = Vals.argsort()[::-1]
    Vals = Vals[I]
    Vecs = Vecs[:,I]
    Vecs = np.dot(D_inv, Vecs)
    Vecs = Vecs / Vecs[0,0]

    Vecs = Vecs[:, 1:nEigenVals]
    Vals = Vals[1:nEigenVals]
    return (Vecs,Vals)

def makeRowStoch(K):
    d = np.sum(K, axis = 0)
    D = np.diag(d)
    D_inv = np.linalg.inv(D)
    return np.dot(D_inv,K)

def ComputeLBAffinity(X, k=16,sig=0):
    Idx, Dx = Knnsearch(X, X, k)
    K,W = ComputeKernel(Idx, Dx, sig=sig)
    d = np.sum(K, axis = 0)
    D = np.diag(d)
    D_inv = np.linalg.inv(D)
    K = np.dot(np.dot(D_inv,K), D_inv)
    return K

def Knnsearch(X,Y,k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    Dx, Idx = nbrs.kneighbors(Y)
    return (Idx,Dx)

def ComputeKernel(Idx,Dx,epsilon=1,sig=0):
    n = Dx.shape[0]
    if sig == 0:
        ep = np.median(Dx,axis=1)
        ep = epsilon*np.tile(ep[:,None],(1,Idx.shape[1]))
    else:
        ep = sig
    temp = np.exp(-np.power(np.divide(Dx,ep),2))
    #temp[np.where(temp<1.0e-3)] = 0
    
    W = np.zeros(shape=(n,n))
    for i in range(n):
        W[i,Idx[i,:]] = temp[i,:]
        
    A = (np.transpose(W) + W)/2  
    return (A,W)
    
