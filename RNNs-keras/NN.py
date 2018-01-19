# RNN IMPLEMENTATION
import pickle
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Activation, Dense
import numpy as np
import time

each_note_size = 89
lowest_note_number = 21
max_time_step = 7

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
  
  
 # config = {num_layers: , hidden_state: []}   
def nn_model(config):
	model = Sequential()
	model.add(LSTM(config['hidden_state'][0], return_sequences=True, input_shape=(max_time_step, each_note_size)))
	model.add(Dropout(0.2))
	for i in range(1,config['num_layers'] - 1):
		model.add(LSTM(config['hidden_state'][i], return_sequences=True))
		model.add(Dropout(0.2))
	if(config['num_layers'] > 2):
		model.add(LSTM(config['hidden_state'][config['num_layers']-1], return_sequences=False))
		model.add(Dropout(0.2))
	model.add(Dense(each_note_size))
	model.add(Activation('softmax'))
	model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
	return model
    
	
def train_nn(X, Y, hm_epochs, config, test):
	model = nn_model(config)
	model.fit(np.array(X), np.array(Y), batch_size=256, epochs=hm_epochs)
	model_json = model.to_json()
	with open("model/model.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("model.h5")
	print("saved model to model/model.json")
    
    
file = open('dataset/Piano-midi.de.pickle', 'rb')
dataset = pickle.load(file)

'''
dataset obtained from: http://www-etud.iro.umontreal.ca/~boulanni/icml2012
note number reference: http://www.electronics.dit.ie/staff/tscarff/Music_technology/midi/midi_note_numbers_for_octaves.htm
Dataset Notes:
database has 3 list train, valid and test
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
n_epochs = 20
batch_size = len(input_train)//200

config = {'num_layers': 3, 'hidden_state': [128, 128, 128], 'n_output': 89, 'n_examples': len(input_train), 'batch_size': batch_size, 'learning_rate': 0.2}

start_time = time.time()
train_nn(input_train, output_train, n_epochs, config, {'input': input_test, 'output': output_test})
print("--- %s seconds ---" % (time.time() - start_time))


"""
Note to self:
    change data input timesteps number
    experiment with different learning rates
    experiment with number of hidden layers and states in each layer
    properly shuffle the data
    [1, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001]
"""



















