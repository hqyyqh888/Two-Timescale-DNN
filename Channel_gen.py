import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import numpy as np
import random
from complex_matrix import *
from Model_singletime import *

Nt = 32  # T antennas
Nr = 16  # R antennas
NtRF = 4  # RF chains at the transmitter
NrRF = 4  # RF chains at the receiver
Ncl = 4   # clusters
Nray = 2  # ray
sigma_h = 0.3  # gain
Tao = 0.001  # delay
fd = 3  # maximum Doppler shift
Batch_size = 32

def theta(N, Seed):
    phi = np.zeros(Batch_size*Ncl*Nray)  # azimuth AoA and AoD
    a = np.zeros((Batch_size, Ncl*Nray, N, 1), dtype=complex)
    np.random.seed(Seed)
    for i in range(Batch_size*Ncl*Nray):
        phi[i] = np.random.uniform(-np.pi/3, np.pi/3)
    f = 0
    for i in range(Batch_size):
        for j in range(Ncl*Nray):
            f += 1
            for z in range(N):
                a[i][j][z] = np.exp(1j * np.pi * z * np.sin(phi[f-1]))
    PHI = phi.reshape(Batch_size, Ncl*Nray)
    #print(a[0][0])
    return a/np.sqrt(N), PHI

def H_gen(Seed):

    HH = torch.zeros((Batch_size, 2, Nr, Nt))
    # random seed
    np.random.seed(Seed)
    # complex gain
    alpha_h = np.random.normal(0, sigma_h, (Batch_size, Ncl*Nray)) + 1j*np.random.normal(0, sigma_h, (Batch_size,Ncl*Nray))
    # receive and transmit array response vectors
    ar, ThetaR = theta(Nr, Seed+10000)
    at, ThetaT = theta(Nt, Seed)
    for b in range(Batch_size):
        H = np.zeros((Nr, Nt), dtype=complex)
        fff = 0
        for i in range(Ncl):
            for j in range(Nray):
                H += alpha_h[b][fff] * np.dot(ar[b][fff], np.conjugate(at[b][fff]).T)
                # H += alpha_h[b][fff] * np.dot(ar[b][fff], np.conjugate(at[b][fff]).T)*np.exp(1j*2*np.pi*Tao*fd*np.cos(ThetaR[b][fff]))    # channel with delay
                fff += 1
        H = np.sqrt(Nt * Nr / Ncl * Nray) * H
        H = c2m(H)
        H = H.to(dtype=torch.float)
        HH[b] = H
    #HH = HH.cuda()
    return HH