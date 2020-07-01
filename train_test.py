import os
import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import math
import sys
import pickle as pkl
import profile_coeffs_generator as pg
import data_generation as dg
import wesnet_model as wm

#parameters
Nt = 30
Nr = 60
snrdb_low_train = 0.0
snrdb_high_train = 14.0
snr_low_train = 10.0 ** (snrdb_low_train/10.0)
snr_high_train = 10.0 ** (snrdb_high_train/10.0)
L = 3*Nt
v_size = 2*Nt
hl_size = 8*Nt
init_LR = 0.0001
decay_factor = 0.97
decay_step_size = 1000
train_epoch = 5
train_samples = 500
train_seed = 213
batch_size = 50
test_iter = 4
test_samples = 500
alpha = 0.9
num_snr = 6
snrdb_low_test = 0.0
snrdb_high_test = 15.0
test_seed = 201



## Weight scaling vector realised with the Linear Profile Functions
L_xl = pg.linear(int(hl_size))
L_vl = pg.linear(int(v_size))

## Weight scaling vector realised with Half-Exponential Profile Function 
L_x = pg.uniform_exp(int(hl_size))
L_v = pg.uniform_exp(int(v_size))

''' 
Compute layer weight scaling vector. You can compute the type weight scaling function you want...
by selecting the type from the above formulations.  
The percentage of input layer weights you want to retain during training and inference
can be selected form 10% - 100% (0.1 - 1.0). 
To obatain BER performance for different percentage of weight scaling, you change the value of weight coefficient
(percentage_idp) to different values, from 0.1 - 1.0.  
'''
profile_x = pg.profile_coefficient(hl_size, L_x, percentage_idp = 0.1, trainable = False)

## Compute the Percentage of Channels Dropped at the Inference Phase
profile_infr = pg.prof_inf(0.1, 1.0, 10, hl_size, L_x)            
len_prof = int(len(profile_infr))

#tensorflow placeholders, the input given to the model in order to train and test the network
HtY = tf.compat.v1.placeholder(tf.float32, shape = [None, Nt])
X = tf.compat.v1.placeholder(tf.float32, shape = [None, Nt])
HtH = tf.compat.v1.placeholder(tf.float32, shape = [None, Nt , Nt])

def train_op(args):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(args)
    return train_step

'''
## A function to calculate the number of model parameter
def count_num_param():
    total_parameters = 0.0
    or variable in tf.trainable_variables():
    local_params = 1
    shape = variable.get_shape()  
    for i in shape:
        local_params *= i.value
    total_parameters += local_params
    return total_parameters
total_parameters = count_num_param()
temp = set(tf.all_variables())
sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
'''

def train(train_samples, profile_x, L, location, train_epoch, abatch_size, Nt, Nr, snr_low_train, snr_high_train, train_seed, regularization = False):
    #tensorflow placeholders, the input given to the model in order to train and test the network
    HtY = tf.compat.v1.placeholder(tf.float32,shape = [None, Nt], name = 'HtY')
    X = tf.compat.v1.placeholder(tf.float32,shape = [None, Nt], name = 'X_in')
    HtH = tf.compat.v1.placeholder(tf.float32,shape = [None, Nt , Nt], name = 'HtH')

    global_step = tf.Variable(0, trainable = False)
    learning_rate = tf.train.exponential_decay(init_LR, global_step, decay_step_size, decay_factor, staircase = True)

    Y_train, H_train, HtY_train, HtH_train, X_train , SNR1 = dg.generate_BPSK_data(train_samples, Nt, Nr, snr_low_train, snr_high_train, train_seed)

    print('HtY: ', HtY_train, '\n')
    print('HtH: ', HtH_train)
    ## Call the WESNET class to build the model 
    network = wm.WESNET(Nt)
    TOTAL_LOSS, LOSS, BER = network(L, X_train, HtY_train, HtH_train, profile_x, regularization = False)
    wesnet_model = (TOTAL_LOSS, LOSS, BER) 

    train_step = train_op(TOTAL_LOSS)  ## training operation object

    ## Initialiser all variables and run the session
    init_g = tf.global_variables_initializer()

    saver = tf.train.Saver()  # create saver object to save the trained model

    with tf.Session() as sess:
        sess.run(init_g)

        #print(total_parameters)

        #Training WeSNet
        tic1 = time.time()
        for epoch in range(train_epoch): #num of train iter 
            for i in range(total_batch):
                idx = np.random.randint(train_samples, size = batch_size) 

                feed_dict = {HY: HY_train[idx, :], 
                   HH: HH_train[idx, :, :], 
                   X: X_train[idx, :]
                   } 

                train_step.run(feed_dict)

            if epoch % 1000 == 0 :
                results = sess.run([loss_ZF, LOSS[L - 1], ber_ZF, BER[L - 1]], feed_dict)
                print_string = [i] + results
                print(' '.join('%s' % x for x in print_string))
        print('Training time: ', tic1 - time.time())

        saved_model = saver.save(sess, location)
  
    #saved_model = wesnetmodel.save(location) 
    return saved_model


'''
## Exponge layers during inference
In our case, since every layer is a prediction layer, we don't have to add a new layer to perform  prediction 
For a conventional deep neural network design, one can specify the number of layers to exponge and add a final prediction layer
'''
def cutting_off_layers(model):

		results = []
		for l in range(len(1, model.layers + 1)):
				if 10 <= l <= len(model.layers):
						x = model.layers[l].output
				results.append(x)
		return results

#Testing the trained model
def test(test_samples, L, model_location, Nt, Nr, snrdb_low_test, snrdb_high_test, num_snr, test_seed, regularization = False):
    tf.reset_default_graph()
    snrdb_list = np.linspace(snrdb_low_test, snrdb_high_test, num_snr)
    snr_list = 10.0 ** (snrdb_list/10.0)
    bers = np.zeros((1, num_snr))
    tmp_ber_iter = np.zeros([L, test_iter])
    ber_iter = np.zeros([L, num_snr])

    '''
    wesnet_model = wm.WESNET(Nt)
    TOTAL_LOSS, LOSS, BER = wesnet_model(L, profile_x, X_train, HtY_train, HtH_train)
    '''

    loss = np.zeros((1, num_snr))
    times = np.zeros((1, num_snr))
    tmp_bers = np.zeros((1, test_iter))
    tmp_loss = np.zeros((1, test_iter))
    tmp_times = np.zeros((1, test_iter))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        restored_model = saver.restore(sess, model_location)

        #restored_model = tf.keras.models.load_wesnetmodel(model_location)

        if regularization:
            results = cutting_off_layers(restored_model)
            TOTAL_LOSS, LOSS, BER = results

        else:
            TOTAL_LOSS, LOSS, BER = restored_model

        tic = time.time()
        for j in range(num_snr):
            for jj in range(test_iter):
                Y_test, H_test, HY_test, HH_test, X_test , SNR_test = dg.generate_BPSK_data(test_samples , Nt, Nr, snr_list[j],snr_list[j], test_seed)

                tmp_bers[0][jj],_ = np.array(sess.run([BER[L - 1], LOSS], {HY: HY_test, HH: HH_test, X: X_test}))
                tmp_ber_iter[:,jj] = np.array(sess.run(BER, {HY: HY_test, HH: HH_test, X: X_test}))
                toc = time.time()
                tmp_times[0][jj] = toc - tic 

            bers[0][j] = np.array(np.mean(tmp_bers, 1))
            times[0][j] = np.array(np.mean(tmp_times[0])/test_batch_size)
            ber_iter[:,j] = np.array(np.mean(tmp_ber_iter, 1))

        print('snrdb_list')
        print(snrdb_list,'\n')
        print('bers')
        print(bers,'\n')
        print('avr_ber for every layer')
        print(ber_iter,'\n')
        print('times')
        print(times,'\n')

        print(ber_iter[1],'\n')
        print(ber_iter[4],'\n')
        print(ber_iter[9],'\n')

        '''
        print(ber_iter[19],'\n')
        print(ber_iter[29],'\n')
        print(ber_iter[39],'\n')
        print(ber_iter[49],'\n')
        print(ber_iter[59],'\n')
        print(ber_iter[69],'\n')
        print(ber_iter[79],'\n')
        print(ber_iter[89],'\n')
        '''
        return 0

model_location = "./WESNET_Model/model_demo.ckpt"

saved_model = train(train_samples, profile_x, L, model_location, train_epoch, batch_size, Nt, Nr, snr_low_train, snr_high_train, train_seed)

restored_model = test(test_samples, model_location, Nt, Nr, snrdb_low_test, snrdb_high_test, num_snr, test_seed)







