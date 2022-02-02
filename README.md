# Two-Timescale-DNN
Two-Timescale End-to-End Learning for Channel Acquisition and Hybrid Precoding

This repository contains the entire code for our work "Two-Timescale End-to-End Learning for Channel Acquisition and Hybrid Precoding", available at: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9610037 and has been accepted for publication in IEEE JOURNAL ON SELECTED AREAS IN COMMUNICATIONS (JSAC).

For any reproduce, further research or development, please kindly cite our JSAC Journal paper:

`Q. Hu, Y. Cai, K. Kang, G. Yu, J. Hoydis, and Y. C. Eldar, "Two-Timescale End-to-End Learning for Channel Acquisition and Hybrid Precoding," IEEE J. Sel. Areas Commun., vol. 40, no. 1, pp. 163-181, Jan. 2022.`

# Requirements
The following versions have been tested: Python 3.6 + Pytorch 1.9.0. But newer versions should also be fine.

## Training and Testing
Firstly, run "`Train_singletime.py`" and save the well-trained model and analog beamformers (set the path at "`torch.save(state, \path)`", "`torch.save(FRF_container, 'path')`", "`torch.save(WRF_container, 'path')`");

Then, run "`Train_twotime.py`" and load the well-trained model and analog beamforming (set the path at "`FRF = torch.load('path')`", "`WRF = torch.load('path')`","`load_data1 = torch.load(path)`", "`load_data2 = torch.load(path)`").


## The introduction of each file
`complex_matrix.py`: Some complex matrix operations;

`Channel_gen.py`: The function of generating channel samples;

`Model_singletime.py`: The model of long-term DNN;

`Model_twotime.py`: The model of short-term DNN;

`Train_singletime.py`: Train long-term DNN;

`Train_twotime.py`: Train short-term DNN.
