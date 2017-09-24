import pickle
import numpy as np

each_note_size = 89
lowest_note_number = 21
max_time_step = 20

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
    return input

 
def getInput(): 
    print("start")    
    with open('dataset/Piano-midi.de.pickle', 'rb') as file:
        print("file open")
        dataset = pickle.load(file)
        print("pickle loaded")
        input_train = np.array(initialize_as_one_hot(np.array(dataset['train'])))
        print("input array made")
        return input_train




















