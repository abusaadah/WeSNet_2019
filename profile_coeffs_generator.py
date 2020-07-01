import numpy as np
import tensorflow as tf
import math

## Create Profile Weights

def uniform(n):
    return [1 for i in range(n)]  ### Uniform weight profile function 

def linear(n, k = 1):
    return [1.0 - (i * k/n) for i in range(n)]  ### Linear weight profile function; equation (16)
            
## Profile Function
def exp(n, k = 1):
    return [np.exp(-k * i) for i in range(n)]

def uniform_exp(n, k=1):              ### Half Exponential weight profile function; equation (17)
    n_half = n // 2
    n_rest = n - n // 2
    return uniform(n_half) + exp(n_rest, k)

def step(n, step = [1]):
    m = len(steps)
    k = 0
    coeffs = []
    for step in steps:
        coeffs.extend([step for x in range(int(np.ceil(1.0 * n/m)))])
        k = k + 1
    return coeffs[: n]

''' We define functions to create profile coefficients vector (beta).
lp = percenatage of the neurons being retained during training and inference
n_neuron = number of neurons in a layer.
weight_coeff_vector = type of the weight profile function.

'''

def nueron_retained(lp, n_neuron, weight_coeff_vector):
    p_L = tf.constant(weight_coeff_vector, shape = [1, n_neuron])
    L_11 = tf.constant(1.0, shape = [1, int(np.round((lp) * n_neuron))])
    L_12 = tf.zeros(shape = [1, int(np.round((1 - lp) * n_neuron))])
    L1 = tf.concat((L_11, L_12), axis = 1)
    p_L1 = tf.multiply(L1, p_L)
    return p_L1

''' 
Genearte Profile Coefficients vector for learnable and non-learnable weight scaling. 
See Sections III A and Section IV C. To make the weight-scaling vector as learnable, 
you set trainable as "True" in the "profile_coefficient" method.
'''

def profile_coefficient(n_neuron, weight_coeff_vec, percentage_idp = None, trainable = None):
    p_l = nueron_retained(percentage_idp, n_neuron, weight_coeff_vec)
    profile_l = tf.stack(p_l, axis = 0) 
    profile_vec = tf.convert_to_tensor(profile_l, dtype = tf.float32)

    if trainable:
        wieght_coeffs_vector = tf.Variable(profile_vec, trainable =  trainable, name = 'weight_scaling_vector')

    else:
        weight_coeffs_vector = tf.Variable(profile_vec, trainable =  None, name = 'weight_scaling_vector')

    return weight_coeffs_vector 
      
## Compute the Percentage of Channels/weights Dropped at the Inference Phase
def prof_inf(min, max, step, n_neuron, weight_coeff_vector):
    profile_infr = []
    percentage_channel = np.linspace(min, max, step)
    for i in percentage_channel:
        p_L1 = nueron_retained(i, n_neuron, weight_coeff_vector)
        profile = tf.stack(p_L1, axis = 0) 
        profile_infr.append(profile)
        profile_i = tf.convert_to_tensor(profile_infr, dtype = tf.float32)
    return profile_infr

print(tf.__version__)