import numpy as np
from midiutil import MIDIFile
from keras.models import model_from_json

lowest_note_number = 21
end_time_step = 88 + lowest_note_number
each_note_size = 89
num_input_note = 7

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
	
	
json_file = open("model/model.json")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")



X = []
X.append(get_note_as_one_hot(109))
X.append(get_note_as_one_hot(32))
X.append(get_note_as_one_hot(42))
X.append(get_note_as_one_hot(52))
X.append(get_note_as_one_hot(62))
X.append(get_note_as_one_hot(72))
X.append(get_note_as_one_hot(109))
'''
X.append(get_note_as_one_hot(67))
X.append(get_note_as_one_hot(end_time_step))
X.append(get_note_as_one_hot(31))
X.append(get_note_as_one_hot(43))
'''
X = np.reshape(X,[1,num_input_note,89])
lst = []
for i in range(num_input_note):
    lst.append(get_note_from_one_hot(X[0][i]))

track    = 0
channel  = 0
time     = 0   # In beats
duration = 1   # In beats
tempo    = 60  # In BPM
volume   = 100 # 0-127, as per the MIDI standard
    
for i in range(7,1000):
	y = model.predict(X)
	note = get_note_from_one_hot(y)    # try sample from a probability distribution
	if i > 8 and (note == lst[i-8] or note == lst[i-7] or note == lst[i-6] or note == lst[i-5] or note == lst[i-4]):
		y[0][np.argmax(y)] = -1
	note = get_note_from_one_hot(y)
	lst.append(note)
	for i in range(num_input_note - 1):
		X[0][i] = X[0][i+1]
	X[0][num_input_note - 1] = get_note_as_one_hot(note)
	# print(get_note_from_one_hot(X[0][0]))
	print(note)
    

MyMIDI = MIDIFile(1, adjust_origin=False)
MyMIDI.addTempo(track,time, tempo)

degree_list = get_note_list(lst)
print(np.shape(degree_list))
for i in range(len(degree_list)):
    for j in range(len(degree_list[i])):
        for pitch in degree_list[i][j]:
            # print(pitch)
            MyMIDI.addNote(track, channel, pitch, time, duration, volume)
            time = time + 1/(12 if (len(degree_list[i][j])) > 12 else (len(degree_list[i][j])))

with open("sample.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)



























