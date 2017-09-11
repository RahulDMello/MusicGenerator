import tensorflow as tf
import numpy as np


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

        for i in range(493):
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
# print(data['features'][0])
# print(data['labels'][0])
# print(data['labels'][492])
total_examples = len(train_data['labels'])

print(np.shape(train_data['features']))

n_nodes_input = 104
n_nodes_hl1 = 52
n_nodes_hl2 = 52
n_nodes_output = 26

x = tf.placeholder('float',[None, 104], name="x")
y = tf.placeholder('float')


def binary_activation(z):
    cond = tf.less(z, tf.multiply(tf.ones(tf.shape(z), dtype=tf.float32), 0.5))
    out = tf.where(cond, tf.zeros(tf.shape(z), dtype=tf.float32), tf.ones(tf.shape(z), dtype=tf.float32))
    
    return out

    
def neural_network_model(data):
    
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_input, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_output])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_output]))}
    
    # (input_data * weights) + biases
    
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    #l1 = binary_activation(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    #l2 = binary_activation(l2)
    
    output = tf.sigmoid(tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases']))
    #output = binary_activation(output)
    output = tf.identity(output, name="output")
    return output



def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.sqrt(tf.reduce_mean((tf.square(tf.subtract(y, prediction)))))
    #                    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 1000
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            _, c = sess.run([optimizer, cost], feed_dict = {x:train_data['features'], y:train_data['labels']})
            epoch_loss = c
            print('Epoch', (epoch+1), 'completed out of', hm_epochs, 'loss: ', epoch_loss)
        
        '''
        # testing part
        correct = tf.equal(prediction, y))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:train_data['features'], y:train_data['labels']}))
        '''
        
        saver.save(sess, 'model/model.ckpt')
        print('saved model at model/model')
        

train_neural_network(x)






















