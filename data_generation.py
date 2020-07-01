import os
import numpy as np
import math


"""Data generation for train and test phases
"""
def generate_BPSK_data(B, Nt, Nr, snr_low, snr_high, seed = 213):
    np.random.seed(seed)
    H = np.random.randn(B, Nr, Nt)
    W = np.zeros([B, Nt, Nt])
    x = np.sign(np.random.rand(B, Nt) - 0.5)
    y = np.zeros([B, Nr])
    n = np.random.randn(B, Nr)
    Hy = x*0
    HH = np.zeros([B, Nt, Nt])
    SNR = np.zeros([B])
    for i in range(B):
      SNR_n = np.random.uniform(low = snr_low, high = snr_high)
      H_n = H[i, :, :]
      snr_factor = (H_n.T.dot(H_n)).trace()/Nt
      H[i, :, :] = H_n
      y[i, :] = (H_n.dot(x[i, :]) + n[i, :] * np.sqrt(snr_factor )/np.sqrt(SNR_n))
      Hy[i, :] = H_n.T.dot(y[i, :])
      HH[i,:,:] = H_n.T.dot( H[i, :, :])
      SNR[i] = SNR_n
    return y, H, Hy, HH, x, SNR

def generate_QPSK_data(B, Nt, Nr, snr_low, snr_high, seed = 201):
    np.random.seed(seed)
    H = np.random.randn(B, Nr, Nt)
    x = np.sign(np.random.rand(B, K) - 0.5)
    y = np.zeros([B, Nr])
    n = np.random.randn(B, Nr)
    Hy = x*0
    HH = np.zeros([B, Nt, Nt])
    SNR = np.zeros([B])
    x_ind = np.zeros([B, Nt, 2])
    for i in range(B):
        for ii in range(Nt):
            if x[i][ii] == 1:
                x_ind[i][ii][0] = 1
            if x[i][ii] == -1:
                x_ind[i][ii][1] = 1   
    for i in range(B):
        SNR_n = np.random.uniform(low = snr_low,high = snr_high)
        H = H[i, :, :]
        tmp_snr = (H.T.dot(H)).trace()/Nt
        H[i, :, :] = H
        y[i, :] = (H.dot(x[i, :]) + n[i, :] * np.sqrt(tmp_snr)/np.sqrt(SNR))
        Hy[i, :] = H.T.dot(y[i, :])
        HH[i, :, :] = H.T.dot( H[i, :, :])
        SNR[i] = SNR_n
    return y, H, Hy, HH, x, SNR, x_ind