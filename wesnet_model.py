import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from numpy import linalg as LA
import math
import sys
import pickle as pkl

'''
This contains the main implementaion of the WeSNet model with sparcity inducing function. 
It also contains codes for the linear, non-linear activation and linear-sof sign (for estimating the received symbols)...
as descibed in equations (8) - (11), (18) and (19).
 
'''


def linear_soft_sign(x):
    t = tf.Variable(0.1)
    y = -1 + tf.nn.relu(x + t)/(tf.abs(t) + 0.00001) - tf.nn.relu(x - t)/(tf.abs(t) + 0.00001)
    return y

class Linear(keras.layers.Layer):
    def __init__(self, input_size, output_size, Layer):
        super(Linear_layer, self).__init__()
        w_init = tf.random_normal_initializer()
        self.W = tf.Variable(initial_value = w_init(shape = (input_size, output_size), dtype = "float32"), 
                  trainable = True, name = 'W'+ Layer)

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape = (output_size,), dtype = "float32"),
                 trainable = True, name = 'b' + Layer)

    def call(self, x):
        y = tf.matmul(inputs, self.W) + self.b

        return y, self.W, self.b

'''
def linear_layer(x, input_size, output_size, Layer):
    W = tf.Variable(tf.random_normal([input_size, output_size], stddev = 0.01), name = 'W'+ Layer)
    b = tf.Variable(tf.random_normal([1, output_size], stddev = 0.01), name = 'b' + Layer)
    y = tf.matmul(x, W) + b
    return y, W, b 
  
'''

def relu_layer(x, input_size, output_size, Layer):
    y, W, b = Linear(input_size, output_size, Layer)(x)
    y = tf.nn.relu(y)
    return y, W, b

def sign_layer(x, input_size, output_size, Layer):
    y, W, b = Linear(input_size, output_size, Layer)(x)
    y = linear_soft_sign(y)
    return y, W, b

''' 
The architecture of WeSNet.
Arguments:
Nt: number of transmit antennas; L: number of layers; X: modulated transmitted symbols;
y: received symbols; H: Fading channel; profile_x: Weight scaling vector.

'''
class WESNET(tf.keras.Model):
    def __init__(self, Nt):
        super(WESNET, self).__init__()
        self.Nt = Nt

    def __call__(self, L, X, HtY, HtH, profile_x, regularization = None):

      train_samples = tf.shape(HtY)[0]

      X_ZF = tf.matmul(tf.expand_dims(HtY, 1), tf.matrix_inverse(HtH))
      X_ZF = tf.squeeze(X_ZF, 1)
      loss_ZF = tf.reduce_mean(tf.square(X - X_ZF))
      ber_ZF = tf.reduce_mean(tf.cast(tf.not_equal(X, tf.sign(X_ZF)), tf.float32))

      v_size = 2 * self.Nt    ## size of the second sub-layer (see table )
      hl_size = 8 * self.Nt  ## size of the first sub-layer

      S = []
      S.append(tf.zeros([train_samples, self.Nt]))
      a = []
      a.append(tf.zeros([train_samples, v_size]))
      LOSS = []
      LOSS.append(tf.zeros([]))
      BER = []
      BER.append(tf.zeros([]))
  
      ## Regularizing coefficients 
      lam_da = tf.Variable(0.001, trainable = True)
      l1_regularizer = tf.contrib.layers.l1_regularizer(scale = 0.005, scope = None)
      weights = tf.trainable_variables() # all vars of your graph
      L1_nrom_reg = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
  
      ## Sublayer weights
      W1r = [] 
      W2r = []
      W3r = []
      for i in range(1, L + 1):
          u_1 = tf.matmul(tf.expand_dims(S[-1], 1), HtH)  
          u = tf.squeeze(u_1, 1)
          x = tf.concat([HtY, S[-1], u, a[-1]], 1)
          Z, W1, b1 = tf.multiply(relu_layer(x, 3*self.Nt + v_size , hl_size,'relu' +str(i)), profile_x) # 1st sub-layer  (18)
          W1r.append(W1)

          a, W2, b2 = linear_layer(tf.multiply(Z, profile_x), hl_size , v_size,'aff'+str(i))  # 2nd sub-layer (19)
          a.append(a)
          W2r.append(W2)
          a[i] = (1 - alpha) * a[i] + alpha * a[i - 1]
    
          S, W3, b3 = sign_layer(tf.multiply(Z, profile_x), hl_size , self.Nt,'sign' +str(i))  # 3rd sub-layer (10)
          S.append(S)
          W3r.append(W3)
          S[i] = (1 - alpha) * S[i] + alpha * S[i - 1]
    
          error_linear = tf.reduce_mean(tf.square(X - X_ZF), 1)   ## Linear receiver error symbol estimated  
          error_wesnet = tf.reduce_mean(tf.square(X - S[-1]), 1)  ## WeSNet symbol error symbol estimate  

          '''
          Sparcity inforcing regularisation for for i = l to L; 
          where l is the layer from which the sparcity is initially enforced. See the paper for the details. 
          '''

          if regularization:
              if 10 <= i <= L + 1 :
                  L1_nrom_reg = tf.norm(W1r[i], ord = 1) + tf.norm(W2r[i], ord = 1) + tf.norm(W3r[i], ord = 1) ## L1 norm regularizer (23)
                  regularization_penalty = lam_da * np.log(1 + (i - 1) * L1_nrom_reg)   ## L1 norm regularizer (22)
                  loss = np.log(i) * tf.reduce_mean(error_wesnet/error_linear) + regularization_penalty ## Regularised loss function (21)
    
          else:
              loss = np.log(i) * tf.reduce_mean(error_wesnet/error_linear)  ## Non-regularised loss function (20)

          LOSS.append(loss)
          BER.append(tf.reduce_mean(tf.cast(tf.not_equal(X,tf.sign(S[-1])), tf.float32)))
          TOTAL_LOSS = tf.add_n(LOSS)
      return TOTAL_LOSS, LOSS, BER

