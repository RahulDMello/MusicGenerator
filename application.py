import tensorflow as tf
import numpy as np


def getNote(y):
    indices = [i for i in range(len(y[0])) if y[0][i] == 1 ] 
    quot = False
    note = ""
    for i in range(len(indices)):
        if(note_representation[indices[i]] == '"'):
            quot = True
        elif(indices[i] == note_representation.index("1") and i < len(indices) - 1):
            ind_1 = note_representation.index("1")
            if(indices[i+1] >= ind_1 and indices[i+1] <= note_representation.index("16")):
                num = [0,0,0,0,0]
                num[indices[i] - ind_1] = 1;
                num[indices[i+1] - ind_1] = 1;
                num.reverse()
                num = int("".join(str(j) for j in num), 2)
                note += str(num)
                i+=1
            else:
                note += note_representation[indices[i]]
        else:
            note += note_representation[indices[i]]

    if(quot):
        note = '"' + note + '"'
    return note


def binary_activation(z):
    cond = np.less(z, np.multiply(np.ones(np.shape(z)), 0.5))
    out = np.where(cond, np.zeros(np.shape(z)), np.ones(np.shape(z)))
    return out


note_representation = ['A','a','B','b','C','c','D','d','E','e','F','f','G','g','Z','z','1','2','4','8','16','"','/','#','.','-']
    
    
sess = tf.Session()
saver = tf.train.import_meta_graph('model/model.ckpt.meta')
saver.restore(sess, 'model/model.ckpt')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
output = graph.get_tensor_by_name("output:0")

# X = [[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]];
# X = [[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
X = [[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]]
for _ in range(60):
    y = sess.run(output, feed_dict = {x: X})
    y = binary_activation(y)
    print(getNote(y))
    del X[0][:26]
    X[0].extend(y[0])



