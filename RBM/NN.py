import pickle
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import data_generator as data

lowest_note = 21
highest_note =108
note_range = highest_note - lowest_note + 2    # 1 for closing time step and 1 because both notes are inclusive
num_timesteps = 20

#This function lets us easily sample from a vector of probabilities
def sample(probs):
    #Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

    
def gibbs_sample(x, k, params):
    #Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(xk):
        hk = sample(visible_to_hidden(xk, params))
        xk = sample(hidden_to_visible(hk, params))
        return hk, xk
        
    x_sample = []
        
    for i in range(k):
        _, x_temp = gibbs_step(x)       # not sure if it will work for k bigger than 1
        x_sample.append(x_temp)
        
    x_sample = tf.stop_gradient(x_sample, name="x_sample")
    
    return x_sample
    

def get_parameters(n_x, n_h):
    W = tf.Variable(tf.random_normal((n_h, n_x)), dtype=tf.float32)
    b_x = tf.Variable(tf.zeros((n_x, 1)))
    b_h = tf.Variable(tf.zeros((n_h, 1)))
    params = {"W": W, "b_x": b_x, "b_h": b_h}
    return params


def visible_to_hidden(x, params):
    W = params["W"]
    b_h = params["b_h"]
    Z = tf.add(tf.matmul(W, x), b_h)
    A = tf.sigmoid(Z)
    return A

def hidden_to_visible(A, params):
    W = params["W"]
    b_x = params["b_x"]
    Z = tf.add(tf.matmul(tf.transpose(W), A), b_x)
    X = tf.sigmoid(Z)
    return X
    
    
def nn_model(x, params, lr):
    # Training Update Code
    # Now we implement the contrastive divergence algorithm. First, we get the samples of x and h from the probability distribution
    #The sample of x
    x_sample = gibbs_sample(x, 1, params) 
    #The sample of the hidden nodes, starting from the visible state of x
    h = sample(visible_to_hidden(x, params)) 
    #The sample of the hidden nodes, starting from the visible state of x_sample
    h_sample = sample(visible_to_hidden(x_sample[0], params)) 

    #Next, we update the values of W, bh, and bv, based on the difference between the samples that we drew and the original values
    size_bt = tf.cast(tf.shape(x)[1], tf.float32)
    W_adder  = tf.multiply(lr/size_bt, tf.subtract(tf.matmul(x, tf.transpose(h)), tf.matmul(x_sample[0], tf.transpose(h_sample))))
    bv_adder = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(x, x_sample[0]), 1, True))
    bh_adder = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 1, True))
    #When we do sess.run(updt), TensorFlow will run all 3 update steps
    updt = [params["W"].assign_add(tf.transpose(W_adder)), params["b_x"].assign_add(bv_adder), params["b_h"].assign_add(bh_adder)]
    return updt

    
def train_model(x, num_epochs, input, lr, num_timesteps, batch_size, n_x, n_h):
    params = get_parameters(n_x, n_h)
    updt = nn_model(x, params, lr)
    with tf.Session() as sess:
    #First, we train the model
    #initialize the variables of the model
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)
        #Run through all of the training data num_epochs times
        print("starting training")
        for epoch in range(num_epochs):
            #Train the RBM on batch_size examples at a time
            for i in range(0, len(input), batch_size): 
                tr_x = input[i:i+batch_size]
                sess.run(updt, feed_dict={x: tr_x.T})
                
            print(epoch+1, "epoch(s) completed out of", num_epochs,"epochs.")
                    
        saver.save(sess, 'model/model')
        print('saved model at model/model')
    
    
    
'''
dataset obtained from: http://www-etud.iro.umontreal.ca/~boulanni/icml2012
note number reference: http://www.electronics.dit.ie/staff/tscarff/Music_technology/midi/midi_note_numbers_for_octaves.htm
Dataset Notes:
datase has 3 list train, valid and test
each list has sequences
each sequence has time steps
each time step has midi note numbers of varying length
note numbers range from [21, 108] for this dataset (in reality they go from 0 to 127)

                # number of notes for one hot    #start and end of a time step        # size of one hot
108 - 21 + 1 =             88                       +                  2                       =            90

i will use 13 note input startup
every time step will be separated using all zeros starting and ending

'''

n_h = 100
n_x = note_range * num_timesteps
x = tf.placeholder(tf.float32, shape=(n_x, None), name="input")
lr = 0.001
input = data.getInput()
input = input.reshape((input.shape[0], -1))
print("input shape: ", input.shape)
num_epochs = 10
num_timesteps = input.shape[1]//note_range
batch_size = input.shape[0]//300
n_x = input.shape[1]
n_h = 100
train_model(x, num_epochs, input, lr, num_timesteps, batch_size, n_x, n_h)



















