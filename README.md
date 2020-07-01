# WeSNet_2019

# Complexity-Scalable Neural Network Based MIMO Detection With Learnable Weight Scaling

# Introduction:
This repository contains the codes for implementing the weight-scaled neural network design for building a low complexity learning-based multiple-inputs multiple-outputs MIMO receivers. The details can be found in our paper: [https://arxiv.org/pdf/1909.06943]

# Prerequisites

To run this code, you will need:
1. Python 3.6 or above
2. TensorFlow 1.15. You can however run the code in TensorFlow 2.xx by disabling the eager execution mode as follows:import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Datasets 
The training and test datasets are generated stochastically from random normal distributions with different instantiations using either BPSK or QPSK modulation (see: data_generation.py). 

# WeSNet Model
The main model is contained in wesnet_model.py. Run wesent_model to compile the model. 

# Training and Testing
The train and test the model, run train_test.py  


