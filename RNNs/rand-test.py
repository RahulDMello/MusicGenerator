
import pickle
import numpy as np
from music21 import *
'''

file = open('dataset/Piano-midi.de.pickle', 'rb')
dataset = pickle.load(file)

print("number of sequences", len(dataset['train']))
print("number of time steps in the first sequence", len(dataset['train'][0]))

for i in range(347):
    if(len(dataset['train'][0][i]) > 5):
        print("notes in the", (i+1) ,"time step of the first sequence", dataset['train'][0][i])

for i in range(87):
    print(len(max(dataset['train'][i],key=len)))



# b = converter.parse('dataset\\Piano-midi\\Piano-midi\\test\\alb_esp2.mid')
# b.show('text')
'''

b = np.ones([3,1,5])
print(b)