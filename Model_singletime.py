import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from complex_matrix import *
from Channel_gen import *
import numpy as np
import torch.nn as nn
import math

L = 16   # pilot length
P = 1    # transmit power
sigma = 0.1  # noise
B = 64  # feedback bits
alpha = 2
flag = 0
Ns = 2     # data stream


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0.0)

# Pilot training
class Pilot(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
        def __init__(self):
            super(Pilot, self).__init__()
            self.X = nn.Parameter(torch.zeros(2, Nt, L))
            # transmit power constraint
            torch.nn.init.normal_(self.X[0], mean=0, std=np.sqrt(P/Nt))
            torch.nn.init.normal_(self.X[1], mean=0, std=np.sqrt(P/Nt))
        def forward(self, H):
            Y = torch.zeros((Batch_size, 2, Nr, L))
            for i in range(Batch_size):
                N = torch.zeros((2, Nr, L), dtype=torch.float)
                N[0] = torch.randn(Nr, L)
                N[1] = torch.randn(Nr, L)
                y = cmul(H[i], self.X) + N
                Y[i, :, :, :] = y
            return Y

# Sigmoid-adjusted ST
class Sigm_Adjust(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
        def __init__(self):
            super(Sigm_Adjust, self).__init__()

        def forward(self, u, Iteration):
            global alpha, flag
            if(Iteration > flag):
                alpha = alpha + 0.2
            flag = Iteration
            y = 2 / (1 + torch.exp(-1*alpha*u)) - 1
            return y

# Sigmoid function
class SigmU(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
        def __init__(self):
            super(SigmU, self).__init__()

        def forward(self, u):
            y = (2 / (1 + torch.exp(-1*u)) - 1)
            return y

class SigmV(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
        def __init__(self):
            super(SigmV, self).__init__()

        def forward(self, u):
            y = 0.15*(2 / (1 + torch.exp(-1*u)) - 1)
            return y

# limit the output to (-pi. pi)
class SigmPHI(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
        def __init__(self):
            super(SigmPHI, self).__init__()

        def forward(self, u):
            y = np.pi*(2 / (1 + torch.exp(-1*u)) - 1)
            return y

class SigmS(nn.Module):
    def __init__(self):
        super(SigmS, self).__init__()

    def forward(self, u):
        y = 1*(2 / (1 + torch.exp(-1 * 3 * u)) - 1)
        return y

# limit the output to 0/1
class SigmBit(nn.Module):
    def __init__(self):
        super(SigmBit, self).__init__()

    def forward(self, u):
        y = 1 / (1 + torch.exp(-1 * 10 * u))
        return y

# CSI feedback
class Feedback(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
        def __init__(self):
            super(Feedback, self).__init__()
            self.layer1 = nn.Sequential(nn.Linear(2 * Nr * L, 512), nn.BatchNorm1d(512), nn.ReLU(True))
            self.layer2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True))
            self.layer3 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True))
            self.layer4 = nn.Sequential(nn.Linear(128, B), nn.BatchNorm1d(B))
            self.layer5 = Sigm_Adjust()
            self.layer6 = SigmBit()

        def forward(self, x, Iteration):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            Q = self.layer4(out)
            Q = self.layer5(Q, Iteration)
            Q = self.layer6(Q)
            return Q

# Recover CSI
class RecoverChannel(nn.Module):
    def __init__(self):
        super(RecoverChannel, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(B, 512), nn.BatchNorm1d(512), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(128, 256), nn.BatchNorm1d(256))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        Hhat = self.layer4(out)
        return Hhat

# AP-NN
class PHI_FRF(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
        def __init__(self):
            super(PHI_FRF, self).__init__()
            self.layer1 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True))
            self.layer2 = nn.Sequential(nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(True))
            self.layer3 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True))
            self.layer4 = nn.Sequential(nn.Linear(128, Nt * NtRF), nn.BatchNorm1d(Nt * NtRF))
            self.layer5 = SigmPHI()

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            PHI_P = self.layer5(out)
            return PHI_P

# AC-NN
class PHI_WRF(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
        def __init__(self):
            super(PHI_WRF, self).__init__()
            self.layer1 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True))
            self.layer2 = nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(True))
            self.layer3 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(True))
            self.layer4 = nn.Sequential(nn.Linear(64, Nr * NrRF), nn.BatchNorm1d(Nr * NrRF))
            self.layer5 = SigmPHI()

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            PHI_C = self.layer5(out)
            return PHI_C

# DP-NN
class FBB(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
        def __init__(self):
            super(FBB, self).__init__()
            self.layer1 = nn.Sequential(nn.Linear(2 * NrRF * NtRF, 50), nn.BatchNorm1d(50), nn.ReLU(True))
            self.layer2 = nn.Sequential(nn.Linear(50, 100), nn.BatchNorm1d(100), nn.ReLU(True))
            self.layer3 = nn.Sequential(nn.Linear(100, 50), nn.BatchNorm1d(50), nn.ReLU(True))
            self.layer4 = nn.Sequential(nn.Linear(50, 2 * NtRF * Ns), nn.BatchNorm1d(2 * NtRF * Ns))
            self.layer5 = SigmU()

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            FBB = self.layer4(out)
            FBB = self.layer5(FBB)
            # print(V)
            return FBB

# DC-NN
class WBB(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
        def __init__(self):
            super(WBB, self).__init__()
            self.layer1 = nn.Sequential(nn.Linear(2 * NrRF * NtRF, 50), nn.BatchNorm1d(50), nn.ReLU(True))
            self.layer2 = nn.Sequential(nn.Linear(50, 100), nn.BatchNorm1d(100), nn.ReLU(True))
            self.layer3 = nn.Sequential(nn.Linear(100, 50), nn.BatchNorm1d(50), nn.ReLU(True))
            self.layer4 = nn.Sequential(nn.Linear(50, 2 * NrRF * Ns), nn.BatchNorm1d(2 * NrRF * Ns))
            self.layer5 = SigmV()

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            WBB = self.layer4(out)
            WBB = self.layer5(WBB)
            return WBB

# Normalize
class NormalizeF(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
        def __init__(self):
            super(NormalizeF, self).__init__()

        def forward(self, FRF, Fbb):
            FBB = torch.zeros((Batch_size, 2, NtRF, Ns))
            for i in range(Batch_size):
                F = torch.norm(cmul(FRF[i], Fbb[i]))
                FBB[i] = math.sqrt(Ns) * (Fbb[i]/F)
            return FBB


def ReceiveS(FBB, FRF, WBB, WRF, H, S, N):
    Shat = torch.zeros((Batch_size, 1, 2, Ns, 1))
    S1 = torch.zeros((Batch_size, 1, 2, Ns, 1))
    Noise = torch.zeros((Batch_size, 1, 2, Ns, 1))  # Noise
    for i in range(Batch_size):
        for j in range(1):
            S1[i][j] = math.sqrt(P)*cmul(cmul(cmul(cmul(cmul(conjT(WBB[i]), conjT(WRF[i])), H[i]), FRF[i]), FBB[i]), S[i][j])
            Noise[i][j] = cmul(cmul(conjT(WBB[i]), conjT(WRF[i])), N[i])
            Shat[i][j] = S1[i][j] + Noise[i][j]
    return Shat

# Demodulation NN
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(2*Ns, 8), nn.BatchNorm1d(8), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(16, 8), nn.BatchNorm1d(8), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(8, 2*Ns), nn.BatchNorm1d(2*Ns))
        self.sigm = SigmS()
    def forward(self, shat):
        out = self.layer1(shat)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.sigm(out)
        return out

class Net(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
        def __init__(self):
            super(Net, self).__init__()
            self.PilotTrain = Pilot()   # pilot training NN
            self.ChannelFeedback = Feedback()  # CSI feedback
            self.RecoverChannel = RecoverChannel()   # Recover CSI
            self.FRFDesign = PHI_FRF()   # AP-NN
            self.WRFDesign = PHI_WRF()   # AC-NN
            self.FBBDesign = FBB()    # DP-NN
            self.WBBDesign = WBB()    # DC-NN
            self.Normalize = NormalizeF()
            self.SigmS = SigmS()
            self.NN = NN()     # Demodulation NN
            # Initialize the weight of the NN
            self.ChannelFeedback.apply(weights_init)
            self.RecoverChannel.apply(weights_init)
            self.WBBDesign.apply(weights_init)
            self.WRFDesign.apply(weights_init)
            self.FBBDesign.apply(weights_init)
            self.FRFDesign.apply(weights_init)
            self.NN.apply(weights_init)
            torch.nn.utils.weight_norm(Pilot(), name='X')
        def forward(self, H, S, N, Iteration):
            Y = self.PilotTrain(H)
            y = torch.zeros((Batch_size, 2 * Nr * L))
            for i in range(Batch_size):
                y[i] = Y[i].reshape(2 * Nr * L)
            # Feedback of channel estimation
            Q = self.ChannelFeedback(y, Iteration)
            # Recover channel
            h_hat = self.RecoverChannel(Q)
            # Analog precoder and combiner design
            PHI_P = self.FRFDesign(h_hat)
            PHI_C = self.WRFDesign(h_hat)
            FRF = torch.zeros((Batch_size, 2, Nt, NtRF))
            WRF = torch.zeros((Batch_size, 2, Nr, NrRF))
            # Constant modulus constraint
            for i in range(Batch_size):
                FRF[i][0] = 1.0/math.sqrt(Nt) * torch.cos(torch.reshape(PHI_P[i], (Nt, NtRF)))
                FRF[i][1] = 1.0/math.sqrt(Nt) * torch.sin(torch.reshape(PHI_P[i], (Nt, NtRF)))
                WRF[i][0] = 1.0/math.sqrt(Nr) * torch.cos(torch.reshape(PHI_C[i], (Nr, NrRF)))
                WRF[i][1] = 1.0/math.sqrt(Nr) * torch.sin(torch.reshape(PHI_C[i], (Nr, NrRF)))
            # Obtain the equivalent channel
            HEQ = torch.zeros((Batch_size, 2, NrRF, NtRF))
            heq = torch.zeros((Batch_size, 2 * NrRF * NtRF))
            for i in range(Batch_size):
                HEQ[i] = cmul(cmul(conjT(WRF[i]), H[i]), FRF[i])
                heq[i] = HEQ[i].reshape(2 * NrRF * NtRF)
            # Digital precoder and combiner design
            fbb = self.FBBDesign(heq)
            wbb = self.WBBDesign(heq)
            Fbb = torch.zeros((Batch_size, 2, NtRF, Ns))
            WBB = torch.zeros((Batch_size, 2, NrRF, Ns))
            for i in range(Batch_size):
                Fbb[i] = torch.reshape(fbb[i], (2, NtRF, Ns))
                WBB[i] = torch.reshape(wbb[i], (2, NrRF, Ns))
            # Normalize the digital precoder to satisfy the power constraint
            FBB = self.Normalize(FRF, Fbb)
            # Obtain the received data
            Shat = ReceiveS(FBB, FRF, WBB, WRF, H, S, N)
            S_Receive = torch.zeros((Batch_size, 1, 2, Ns, 1))
            shat = torch.zeros((Batch_size, 1, 2 * Ns * 1))
            s_receive = torch.zeros((Batch_size, 1, 2 * Ns * 1))
            for i in range(Batch_size):
                for j in range(1):
                    shat[i][j] = Shat[i][j].reshape(2 * Ns * 1)
            # Demodulation
            for i in range(1):
                s_receive[:, i, :] = self.NN(shat[:, i, :])
            for i in range(Batch_size):
                for j in range(1):
                    S_Receive[i][j] = torch.reshape(s_receive[i][j], (2, Ns, 1))

            return S_Receive, FRF, WRF
