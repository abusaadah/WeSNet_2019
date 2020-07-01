# WeSNet_2019
# Complexity-Scalable Neural Network Based MIMODetection With Learnable Weight Scaling.

# Introduction
This repository contains the implementation of the weight-scaled deep neural network (WeSNet) for designing a learning-based low complexity Multiple-Inputs Multiple-Outputs (MIMO) receivers. We refer the reader to our paper for details:

# Prerequisites
To run the code in this code in this repository, you will need:

1. Python 3.68 or higher version
2. Tensorflow 1.15 or higher

# Datasets
The datasets are contained in the file: "data_generation.py". Training and Test datasets are generated stochastically from normal distributions. For the transmitted symbols, BPSK and QPSK or QAM modulations are used (see details in the paper).

# Training and Testing

Run the wesnet_model.py to compile the WeSNet model. For training and inference, run the train_test.py.
To obtain BER performance for different percentages of weight scaling, you change the value of weight coefficient
to different values, from 0.1 - 1.0. If you are using Tensorflow 2.xx, turn off the eager execution before running the code.


