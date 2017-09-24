import tensorflow as tf
import numpy as np
from midiutil import MIDIFile

lowest_note_number = 21
end_time_step = 88 + lowest_note_number
each_note_size = 89
num_input_note = 20

def get_note_as_one_hot(data):
    temp = np.array(data)
    temp = temp - lowest_note_number
    b = np.zeros([1, each_note_size], np.int)
    b[np.arange(1), temp] = 1
    return b
    
def get_note_from_one_hot(note):
    return np.argmax(note) + lowest_note_number
    
def get_note_list(notes):
    lst = []
    temp = []
    for note in notes:
        if(note == end_time_step and temp):
            lst.append([temp])
            temp = []
        else:
            temp.append(note)
    if(temp):
        lst.append([temp])
    return lst

    
sess = tf.Session()
saver = tf.train.import_meta_graph('model/model.meta')
saver.restore(sess, 'model/model')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("input:0")
x_sample = graph.get_tensor_by_name("x_sample:0")

X = []
X.extend(get_note_as_one_hot(36))
X.extend(get_note_as_one_hot(48))
X.extend(get_note_as_one_hot(end_time_step))
X.extend(get_note_as_one_hot(36))
X.extend(get_note_as_one_hot(48))
X.extend(get_note_as_one_hot(60))
X.extend(get_note_as_one_hot(64))
X.extend(get_note_as_one_hot(67))
X.extend(get_note_as_one_hot(end_time_step))
X.extend(get_note_as_one_hot(31))
X.extend(get_note_as_one_hot(43))
X.extend(get_note_as_one_hot(60))
X.extend(get_note_as_one_hot(64))
X.extend(get_note_as_one_hot(67))
X.extend(get_note_as_one_hot(end_time_step))
X.extend(get_note_as_one_hot(36))
X.extend(get_note_as_one_hot(48))
X.extend(get_note_as_one_hot(end_time_step))
X.extend(get_note_as_one_hot(36))
X.extend(get_note_as_one_hot(48))
X = np.array(X)
X = X.reshape([-1, 1])
print("shape of x", np.shape(X))

lst = []

track    = 0
channel  = 0
time     = 0   # In beats
duration = 1   # In beats
tempo    = 60  # In BPM
volume   = 100 # 0-127, as per the MIDI standard
    
    
y = sess.run(x_sample, feed_dict = {x: X})
y = np.array(y)
y = y.reshape((y.shape[1], 1))
print("shape of y", y.shape)
for i in range(0,y.shape[0],each_note_size):
    note = get_note_from_one_hot(y[i: i+each_note_size])    # try sample from a probability distribution
    lst.append(note)
    # print(get_note_from_one_hot(X[0][0]))
    print("note", note)
    
print("shape of lst", np.shape(lst))
    

MyMIDI = MIDIFile(1, adjust_origin=False)
MyMIDI.addTempo(track,time, tempo)

degree_list = get_note_list(lst)
print("degree list", np.shape(degree_list))
for i in range(len(degree_list)):
    for j in range(len(degree_list[i])):
        for pitch in degree_list[i][j]:
            # print(pitch)
            MyMIDI.addNote(track, channel, pitch, time, duration, volume)
            time = time + 1/(4 if (len(degree_list[i][j])) > 4 else (len(degree_list[i][j])))

with open("sample.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)



























