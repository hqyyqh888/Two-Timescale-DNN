import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from complex_matrix import *
from Model_twotime import *
from Channel_gen import *
import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
batch = 100
Epoch = 100
seed = [1+1j, 1-1j, -1+1j, -1-1j]

FRF_container = torch.zeros((Batch_size, 2, Nt, NtRF))
WRF_container = torch.zeros((Batch_size, 2, Nr, NrRF))

def Produce_Data(Seed):
    # Noise
    torch.manual_seed(Seed)
    N = torch.normal(mean=torch.full((Batch_size * 2 * Nr * 1, 1), 0.0), std=torch.full((Batch_size * 2 * Nr * 1, 1), sigma))
    NN = torch.reshape(N, (Batch_size, 2, Nr, 1))

    # signal
    np.random.seed(Seed)
    S = np.random.choice(seed, size=1*Ns*Batch_size)
    S = c2m(S)
    S = S.to(dtype=torch.float)
    SS = torch.reshape(S, (Batch_size, 1, 2, Ns, 1))

    return SS, NN


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1 = Net1()        # long-term DNN
model2 = Net2()        # short-term DNN
'''
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model1 = nn.DataParallel(model1)
    model2 = nn.DataParallel(model2)
model1 = model1.to(device)
model2 = model2.to(device)
'''
# load model of long-term DNN and short-term DNN
# load_data1 = torch.load(path)
# load_data2 = torch.load(path)
# model1.load_state_dict(load_data1['state_dict'])
# model2.load_state_dict(load_data2['state_dict'])

# load FRF and WRF obtained by long-term DNN
# FRF = torch.load('path')
# WRF = torch.load('path')
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
#optimizer.load_state_dict(load_data['optimizer'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.98)

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch >= 0 and epoch < 6:
        lr = 0.01
    elif epoch >= 6 and epoch < 10:
        lr = 0.001
    elif epoch >= 10 and epoch < 100:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

loss = 0
Loss = 0
mse = 0
Mse = []
Mse_test = []
I = []
T = 20000
Heq = torch.zeros((Batch_size, 2, NrRF, NtRF))
Neq = torch.zeros((Batch_size, 2, NrRF, 1))
Heq_test = torch.zeros((Batch_size, 2, NrRF, NtRF))
Neq_test = torch.zeros((Batch_size, 2, NrRF, 1))
SEED = np.zeros((batch), dtype=int)
for i in range(batch):
    SEED[i] = i

lr_init = optimizer2.param_groups[0]['lr']
model11 = model1.eval()
model22 = model2.eval()
for epoch in range(Epoch):
    STEP = 0
    print("Iteration:", epoch)
    adjust_learning_rate(optimizer2, epoch, lr_init)
    lr = optimizer2.param_groups[0]['lr']
    print(epoch, lr)
    if epoch % 10 == 0:
        SEED += batch
    for step in range(batch):
        mse = 0
        Signal, Noise = Produce_Data(SEED[step])  # Obtain signal and noise for training
        channel = H_gen(SEED[step])  # Obtain channel data for training
        # Train
        Shat = model2(channel, Signal, Noise, FRF[step], WRF[step], epoch)
        # MSE for the loss function
        for m in range(Batch_size):
            for n in range(1):
                mse += (torch.norm(Signal[m][n] - Shat[m][n]) ** 2)
        Loss += (mse / 1) / Batch_size
        mse = (mse / 1) / Batch_size
        print('batch:', step)
        print('mse:', mse)
        optimizer2.zero_grad()
        mse.backward()
        optimizer2.step()
    # scheduler.step()
    print(Loss/batch)
    # save the model of short-term DNN
    state = {
        'epoch': epoch,
        'state_dict': model2.state_dict(),
        'optimizer': optimizer2.state_dict(),
    }
    # torch.save(state, r'path')
    #test
    with torch.no_grad():
        Test_Mse = 0
        for t in range(10):
            Signal_t, Noise_t = Produce_Data(T-t)  # Obtain signal and noise for test
            channel_t = H_gen(T-t)  # Obtain channel data for test
            Shat11, FRF_container, WRF_container = model11(channel_t, Signal_t, Noise_t, epoch)
            Shat_test = model2(channel_t, Signal_t, Noise_t, FRF_container, WRF_container, epoch)
            mse_test = 0
            for mm in range(Batch_size):
                for nn in range(1):
                    mse_test += (torch.norm(Signal_t[mm][nn] - Shat_test[mm][nn]) ** 2).item()
            Test_Mse += (mse_test/1) / Batch_size
        print('Test Mse:', Test_Mse / 10)
