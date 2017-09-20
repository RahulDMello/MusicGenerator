import pickle
import tensorflow as tf
import numpy as np

each_note_size = 89
lowest_note_number = 21
max_time_step = 3

def get_one_hot(data):
    temp = np.array(data)
    temp = temp - lowest_note_number
    temp = [int(v) for v in temp]
    b = np.zeros([len(temp), each_note_size], np.int)
    b[np.arange(len(temp)), temp] = 1
    return b

def get_one_hot_list(data):
    eo_time_step = get_one_hot([109])
    lst = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            b = get_one_hot(data[i][j])
            lst.extend(b)
            lst.extend(eo_time_step)
            
    return lst
    
def collect_input(lst):
    input = []
    temp = []
    curr_time_step = 1
    for i in range(len(lst) - max_time_step):
        temp = []
        for j in range(max_time_step):
            temp.append(lst[i + j])
        input.append(temp)
    return input
    
def initialize_as_one_hot(data):
    lst = get_one_hot_list(data)
    input = collect_input(lst)
    output = lst[max_time_step:]
    return input, output

def seq_length(sequence):
    print("seq_length")
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    print("seq_length")
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length
  
  
 # config = {num_layers: , hidden_state: []}   
def nn_model(data, config):
    cells = []
    for i in range(config['num_layers']):
        cell = tf.nn.rnn_cell.LSTMCell(config['hidden_state'][i])
        cells.append(cell)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    
    
    output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
    
    output = tf.gather(output, output.shape[1]-1, axis=1) 
    
    weight = tf.Variable(tf.truncated_normal([config['hidden_state'][-1], config['n_output']]))
    bias = tf.Variable(tf.zeros([1, config['n_output']]))    # or [1, num_output]
    prediction = tf.matmul(output, weight) + bias    # removed softmax
    return prediction
    
    
def cost_function(target, prediction):
    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
    return cross_Entropy
    
def train_nn(x, y, X, Y, hm_epochs, config, test):
    prediction = nn_model(x, config)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate']).minimize(cost)
    prediction = tf.nn.softmax(prediction, name = "output")
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('starting training.')
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for ind in range(int(config['n_examples']/config['batch_size'])):
                epoch_x, epoch_y = X[ind*batch_size:ind*batch_size+batch_size], Y[ind*batch_size:ind*batch_size+batch_size]

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                
                # print("mini batch number ", ind + 1)

            print('Epoch', epoch + 1, 'completed out of',hm_epochs,'loss:',epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test['input'], y:test['output']}))
        
        saver.save(sess, 'model/model')
        print('saved model at model/model')
    
    
file = open('dataset/Piano-midi.de.pickle', 'rb')
dataset = pickle.load(file)

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
    
# BASIC VARIABLE AND META VARIABLE INITIALIZE
input_train, output_train = initialize_as_one_hot(np.array(dataset['train']))
input_test, output_test = initialize_as_one_hot(np.array(dataset['valid']))
print(np.array(input_train).shape)
print(len(input_train))
print(len(output_train))
print(len(input_test))
print(len(output_test))
'''
print(input_test[:2])
print("output")
print(output_test[:2])
'''
n_epochs = 15
batch_size = len(input_train)//200

config = {'num_layers': 1, 'hidden_state': [128], 'n_output': 89, 'n_examples': len(input_train), 'batch_size': batch_size, 'learning_rate': 0.05}
x  = tf.placeholder('float',[None, max_time_step, 89], name="input")
y = tf.placeholder('float')
train_nn(x, y, input_train, output_train, n_epochs, config, {'input': input_test, 'output': output_test})



"""
Note to self:
    change data input timesteps from 1 to a bigger number
    change the reshape of output in nn_model to support more timesteps
    experiment with different learning rates
    experiment with number of hidden layers and states in each layer
"""



















