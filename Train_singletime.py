import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import numpy as np
import random
from complex_matrix import *
from Model_singletime import *
from Channel_gen import *
import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau

# QPSK
seed = [1+1j, 1-1j, -1+1j, -1-1j]
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
model = Net()
# load the model
# load_data = torch.load(\path)
'''
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model = model.to(device)
'''
#model.load_state_dict(load_data['state_dict'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer.load_state_dict(load_data['optimizer'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
def adjust_learning_rate(optimizer, epoch, lr):
    if epoch >= 0 and epoch < 5:
        lr = 0.01
    elif epoch >= 5 and epoch < 10:
        lr = 0.001
    elif epoch >= 10 and epoch < 100:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


batch = 100
Epoch = 100
T = 100000
# random seed
SEED = np.zeros((batch), dtype=int)
for i in range(batch):
    SEED[i] = i

lr_init = optimizer.param_groups[0]['lr']
FRF_container = torch.zeros(batch, Batch_size, 2, Nt, NtRF)
WRF_container = torch.zeros(batch, Batch_size, 2, Nr, NrRF)
for epoch in range(Epoch):
    print("Iteration:", epoch)
    # adjust lr
    adjust_learning_rate(optimizer, epoch, lr_init)
    lr = optimizer.param_groups[0]['lr']
    print(epoch, lr)
    if epoch % 10 == 0:
        SEED += batch
    Loss = 0
    for step in range(batch):
        Signal, Noise = Produce_Data(SEED[step])  # Obtain signal and noise for training
        channel = H_gen(SEED[step])  # Obtain channel data for training
        # Train
        Signal_hat, FRF, WRF = model(channel, Signal, Noise, epoch)
        FRF_container[step] = FRF
        WRF_container[step] = WRF
        # MSE for the loss function
        mse = 0
        for m in range(Batch_size):
            for n in range(1):
                mse += (torch.norm(Signal[m][n] - Signal_hat[m][n])**2)
        Loss += (mse/1) / Batch_size
        mse = (mse/1) / Batch_size
        print('batch:', step)
        print('mse:', mse)
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()
    print(Loss/batch)
    # save FRF and WRF
    # torch.save(FRF_container, 'path')
    # torch.save(WRF_container, 'path')
    # save the model
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    # torch.save(state, \path)
    # test
    with torch.no_grad():
        Test_Mse = 0
        for t in range(10):
            Stest, Ntest = Produce_Data(T+t)  # Obtain signal and noise for testing
            channel_test = H_gen(T+t)  # Obtain channel data for testing
            Shat_test, FRF, WRF = model(channel_test, Stest, Ntest, epoch)
            mse_test = 0
            for mm in range(Batch_size):
                for nn in range(1):
                    mse_test += (torch.norm(Stest[mm][nn] - Shat_test[mm][nn]) ** 2).item()
            Test_Mse += (mse_test/1) / Batch_size
        print('Test Mse:', Test_Mse / 10)
