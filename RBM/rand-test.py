import pickle
import numpy as np
from midiutil import MIDIFile

    
file = open('dataset/Piano-midi.de.pickle', 'rb')
dataset = pickle.load(file)

print(dataset['test'][0][:5])

'''
degrees  = [60, 62, 64, 65, 67, 69, 71, 72] # MIDI note number
track    = 0
channel  = 0
time     = 0   # In beats
duration = 1   # In beats
tempo    = 60  # In BPM
volume   = 100 # 0-127, as per the MIDI standard

MyMIDI = MIDIFile(1, adjust_origin=False) # One track, defaults to format 1 (tempo track
                     # automatically created)
MyMIDI.addTempo(track,time, tempo)

for i in range(len(dataset['train'][0])):
    for pitch in dataset['train'][0][i]:
        MyMIDI.addNote(track, channel, pitch, time, duration, volume)
        time = time + 1/(2*len(dataset['train'][0][i]))

with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)

'''






