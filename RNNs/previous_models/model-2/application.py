import tensorflow as tf
import numpy as np
from midiutil import MIDIFile

lowest_note_number = 21
end_time_step = 88 + lowest_note_number
each_note_size = 89

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
output = graph.get_tensor_by_name("output:0")

X = []
X.append(get_note_as_one_hot(55))
X.append(get_note_as_one_hot(79))
X.append(get_note_as_one_hot(end_time_step))
X= np.reshape(X,[1,3,89])
lst = [get_note_from_one_hot(X[0][0]), get_note_from_one_hot(X[0][1]), get_note_from_one_hot(X[0][2])]

track    = 0
channel  = 0
time     = 0   # In beats
duration = 1   # In beats
tempo    = 60  # In BPM
volume   = 100 # 0-127, as per the MIDI standard
    
for _ in range(150):
    y = sess.run([output], feed_dict = {x: X})
    note = get_note_from_one_hot(y)
    # print(note)
    lst.append(note)
    X[0][0], X[0][1], X[0][2] = X[0][1], X[0][2], get_note_as_one_hot(note)

MyMIDI = MIDIFile(1, adjust_origin=False)
MyMIDI.addTempo(track,time, tempo)

degree_list = get_note_list(lst)
print(np.shape(degree_list))
for i in range(len(degree_list)):
    for j in range(len(degree_list[i])):
        for pitch in degree_list[i][j]:
            print(pitch)
            MyMIDI.addNote(track, channel, pitch, time, duration, volume)
            time = time + 1/(4 if (2*len(degree_list[i][j])) > 8 else (2*len(degree_list[i][j])))

with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)





























