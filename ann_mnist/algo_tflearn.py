import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def get_dataset_info(train):
    print(train.shape)
    # print(train.describe())

def one_hot_data(y):
    n_classes = len(np.unique(y))
    hot_array_y = np.zeros([len(y),n_classes])
    for i in range(len(y)):
        j=y[i]
        hot_array_y[i,j]=1

    return hot_array_y



train=pd.read_csv('train.csv',nrows=1000)
# print("Training data information : "  + get_dataset_info(train) )

# training dataset
x_train=train.iloc[:,1:].values.astype('float32')
y_train=train.iloc[:,0].values
y_train=one_hot_data(y_train)

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.25,random_state=1)

# model variables
x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

# model parameters
learning_rate,batch_size=0.001,100
layer1_nodes,layer2_nodes,layer3_nodes=100,100,100
output_layer_nodes=10

def neural_network_graph(data):

    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784,layer1_nodes])),
                      'bias':tf.Variable(tf.random_normal([layer1_nodes]))}
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([layer1_nodes, layer2_nodes])),
                      'bias': tf.Variable(tf.random_normal([layer2_nodes]))}
    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([layer2_nodes, layer3_nodes])),
                      'bias': tf.Variable(tf.random_normal([layer3_nodes]))}
    output_layer =  {'weights': tf.Variable(tf.random_normal([layer3_nodes, output_layer_nodes])),
                      'bias': tf.Variable(tf.random_normal([output_layer_nodes]))}


    # layer1 function
    l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,output_layer['weights']),output_layer['bias'])
    return output


def neural_network_computation():

    prediction = neural_network_graph(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer=tf.train.AdamOptimizer().minimize(cost)

    total_epoch=500
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(total_epoch):
            epoch_loss=0
            for i in range(int(len(x_train)/batch_size)):

                epoch_x = x_train[i*100:i*100+batch_size,:]
                epoch_y = y_train[i * 100:i * 100 + batch_size, :]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch ',epoch,'completed out of ',total_epoch,'loss ',epoch_loss)

        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy = ', accuracy.eval({x:x_test,y:y_test}))

neural_network_computation()