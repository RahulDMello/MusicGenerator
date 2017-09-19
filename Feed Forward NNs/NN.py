import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


#filename is an array ie [filename]
def getData(filename):
    X = []
    Y = []
    filename_queue = tf.train.string_input_producer(filename)

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [tf.constant([], dtype=tf.string),    # Column 0
                       tf.constant([], dtype=tf.string),    # Column 1
                       tf.constant([], dtype=tf.string),    # Column 2
                       tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]
    col1, col2, col3, col4, col5 = tf.decode_csv(
        value, record_defaults=record_defaults)
    features = tf.stack([col1, col2, col3, col4])

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1369):
            # Retrieve a single instance:
            single_x, label = sess.run([features, col5])
            single_x = [[int(single_x[0][i:i+1]) for i in range(0, len(single_x[0]), 1)]+
                         [int(single_x[1][i:i+1]) for i in range(0, len(single_x[0]), 1)]+
                         [int(single_x[2][i:i+1]) for i in range(0, len(single_x[0]), 1)]+
                         [int(single_x[3][i:i+1]) for i in range(0, len(single_x[0]), 1)]]
            label = [[label[i:i+1] for i in range(0, len(label), 1)]]
            X.extend(single_x)
            Y.extend(label)
        coord.request_stop()
        coord.join(threads)
        
    return {'features': X, 'labels': Y}

    

train_data = getData(["data/data.csv"])
train_data['features'], train_data['labels'] = shuffle(train_data['features'], train_data['labels'])
# print(data['features'][0])
# print(data['labels'][0])
# print(data['labels'][492])
total_examples = len(train_data['labels'])

# print(np.shape(train_data['features']))

n_nodes_input = 104
n_nodes_hl1 = 52
n_nodes_hl2 = 52
n_nodes_hl3 = 26
n_nodes_output = 26

x = tf.placeholder('float',[None, 104], name="x")
y = tf.placeholder('float')
randomize = tf.placeholder('int32', name="randomize")


def binary_activation(z):
    cond = tf.less(z, tf.multiply(tf.ones(tf.shape(z), dtype=tf.float32), 0.5))
    out = tf.where(cond, tf.zeros(tf.shape(z), dtype=tf.float32), tf.ones(tf.shape(z), dtype=tf.float32))
    
    return out

    
def neural_network_model(data):
    
    n_nodes_input_div = int(n_nodes_input/4)
    n_nodes_h1_div = int(n_nodes_hl1/4)
    
    hidden_1_layer_1 = {'weights': tf.Variable(tf.random_normal([n_nodes_input_div, n_nodes_h1_div])),
                      'biases': tf.Variable(tf.zeros([1,n_nodes_h1_div]))}
    hidden_1_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_input_div, n_nodes_h1_div])),
                      'biases': tf.Variable(tf.zeros([1,n_nodes_h1_div]))}
    hidden_1_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_input_div, n_nodes_h1_div])),
                      'biases': tf.Variable(tf.zeros([1,n_nodes_h1_div]))}
    hidden_1_layer_4 = {'weights': tf.Variable(tf.random_normal([n_nodes_input_div, n_nodes_h1_div])),
                      'biases': tf.Variable(tf.zeros([1,n_nodes_h1_div]))}
                      
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.zeros([1,n_nodes_hl2]))}
    
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                    'biases': tf.Variable(tf.zeros([1,n_nodes_hl3]))}
    
    def f1(): 
        return tf.abs(tf.subtract(tf.random_uniform(tf.shape(data), minval=0.2, maxval=0.3, dtype=tf.float32), data))
    def f2(): return data
    data = tf.cond(tf.equal(randomize, tf.constant(0), name="check"), f2, f1)
    
    data_1 = tf.slice(data, [0,0], [tf.shape(data)[0],26])
    data_2 = tf.slice(data, [0,26], [tf.shape(data)[0],26])
    data_3 = tf.slice(data, [0,26*2], [tf.shape(data)[0],26])
    data_4 = tf.slice(data, [0,26*3], [tf.shape(data)[0],26])
    
    l1_1 = tf.nn.relu(tf.add(tf.matmul(data_1, hidden_1_layer_1['weights']), hidden_1_layer_1['biases']))
    l1_2 = tf.nn.relu(tf.add(tf.matmul(data_2, hidden_1_layer_2['weights']), hidden_1_layer_2['biases']))
    l1_3 = tf.nn.relu(tf.add(tf.matmul(data_3, hidden_1_layer_3['weights']), hidden_1_layer_3['biases']))
    l1_4 = tf.nn.relu(tf.add(tf.matmul(data_4, hidden_1_layer_4['weights']), hidden_1_layer_4['biases']))
    
    # (input_data * weights) + biases
    
    l1 = tf.concat([l1_1, l1_2, l1_3, l1_4], 1)
    #l1 = binary_activation(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    #l2 = binary_activation(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output_1 = tf.slice(l3, [0,0], [tf.shape(l3)[0],16])
    output_2_3 = tf.slice(l3, [0,0], [tf.shape(l3)[0],10])
    
    #output = binary_activation(output)
    ac_output = tf.concat([tf.nn.softmax(output_1),tf.sigmoid(output_2_3)],1, name="output")
    return output_1, output_2_3
    
    
def cost_function_1(target, prediction):
    output_1 =tf.slice(target,[0,0],[tf.shape(target)[0], 16])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_1, logits=prediction))
    return cost
    
def cost_function_2_3(target, prediction):
    output_2_3 = tf.slice(target,[0,16],[tf.shape(target)[0], 10])
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output_2_3, logits=prediction))
    cost = tf.reduce_mean(cost)
    return cost
    
def avg_distance_between(p_1, p_2):
    #tf.Print(p_2, p_2)
    res = tf.subtract(p_1, p_2)
    res = tf.square(res)
    res = tf.reduce_sum(res, 1)
    res = tf.sqrt(res)
    res = tf.reduce_mean(res)
    return res
    


def train_neural_network(x):
    pred_1, pred_2_3 = neural_network_model(x)
    cost_1 = cost_function_1(y, pred_1)
    cost_2_3 = cost_function_2_3(y, pred_2_3)
    #                    learning_rate = 0.001
    optimizer_1 = tf.train.AdamOptimizer().minimize(cost_1)
    optimizer_2_3 = tf.train.AdamOptimizer().minimize(cost_2_3)
    
    hm_epochs = 250
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        for epoch in range(hm_epochs):
            
            epoch_loss = 0
            _, c1 = sess.run([optimizer_1, cost_1], feed_dict = {x:train_data['features'], y:train_data['labels'], randomize:1})
            epoch_loss += c1
            _, c23 = sess.run([optimizer_2_3, cost_2_3], feed_dict = {x:train_data['features'], y:train_data['labels'], randomize:1})
            epoch_loss += c23
            print('Epoch', (epoch+1), 'completed out of', hm_epochs, 'loss: (',c1,'+',c23,') ', epoch_loss)
        
        
        # testing part
        correct = avg_distance_between(tf.concat([pred_1, pred_2_3], 1), y)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:train_data['features'], y:train_data['labels'], randomize:1}))
        
        
        saver.save(sess, 'model/model.ckpt')
        print('saved model at model/model')
        

train_neural_network(x)






















