import tensorflow as tf
import numpy as np


def getNote(y):
    ind_1 = note_representation.index("1")
    indices = [i for i in range(len(y[0])) if y[0][i] == 1 ] 
    quot = False
    hasNum = False
    numArr = []
    note = ""
    for i in range(len(indices)):
        if(note_representation[indices[i]] == '"'):
            quot = True
        elif(indices[i] >= note_representation.index("1") and  indices[i] <= note_representation.index("16")):
            hasNum = True
            numArr.append(indices[i])
        else:
            note += note_representation[indices[i]]
            
    if(hasNum):
        num = [0,0,0,0,0]
        for i in numArr:
            num[i - ind_1] = 1;
            
        num.reverse()
        num = int("".join(str(j) for j in num), 2)
        note += str(num) #'' if num > 8 else num)
        
    if(quot):
        note = '"' + note + '"'
    return note


def sigmoid_binary_activation(z):
    common_threshold = 0.6
    cond = np.less(z, np.ones(np.shape(z)) * common_threshold)
    out = np.where(cond, np.zeros(np.shape(z)), np.ones(np.shape(z)))
    return out
    
def max_of_softmax(z):
    b = np.zeros_like(z)
    b[z.argmax()] = 1
    return b
    
def sigmoid_threshold_binary_activation(z, threshold):
    if(z.max() > threshold):
        # print(z.max())
        return max_of_softmax(z)
    else:
        return np.zeros_like(z)


note_representation = ['A','a','B','b','C','c','D','d','E','e','F','f','G','g','Z','z','#','.','-','/','1','2','4','8','16','"']
    
    
sess = tf.Session()
saver = tf.train.import_meta_graph('model/model.ckpt.meta')
saver.restore(sess, 'model/model.ckpt')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
output = graph.get_tensor_by_name("output:0")
randomize = graph.get_tensor_by_name("randomize:0")
check = graph.get_tensor_by_name("check:0")

previous_notes= [None, None]



# X = [[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]];
# X = [[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
# X = [[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]]
X = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0]]

# X = [0] * 104
# X = [X]

# GABc
# X = [[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

sampleFile = open("sample.abc","w")
sampleFile.write("L:1/4\nM:4/4\nK:G\n\n")
sampleFile.write(getNote([X[0][0:26]]) + " ")
sampleFile.write(getNote([X[0][26:26*2]]) + " ")
sampleFile.write(getNote([X[0][26*2:26*3]]) + " ")
sampleFile.write(getNote([X[0][26*3:]]) + " ")
for _ in range(150):
    y, checker = sess.run([output, check], feed_dict = {x: X, randomize: 1})
    z = [0] * len(y[0])
    z = [z]
    z[0][19:] = sigmoid_binary_activation(y[0][19:])
    #                                                                                   # decide this parameter
    z[0][16:19] = sigmoid_threshold_binary_activation(y[0][16:19], 0.09)
    z[0][:16] = max_of_softmax(y[0][:16])
    note = getNote(z)
    
    
    if note == previous_notes[0] or note == previous_notes[1]:
        print("repeated note: ",note)
        print("argmax: ", y[0][:16].argmax())
        print("min: ", y[0][:16].min())
        y[0][y[0][:16].argmax()] = y[0][:16].min()
        z[0][:16] = max_of_softmax(y[0][:16])
        note = getNote(z)
    
    
    previous_notes[0], previous_notes[1] = previous_notes[1], note
    # print(note)
    print(checker)
    sampleFile.write(note + " ")
    del X[0][:26]
    X[0].extend(z[0])

#print("check = ",check)
sampleFile.write("\n")
print("midi exported to sample.abc")
sampleFile.close()
    


