#!/usr/bin/env python
# coding: utf-8

#----Import Packages----#
from struct import *
import os
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


#----Params-----#
# Translate the values to positive
translate = True

#Neural Network
n_hidden = 60 # number of neurons of hidden layer
learning_rate = 0.001 
n_epochs = 10000 # number of epochs
batch_size = 500 # size of the mini batch
dropout_prob = 0.5

n_training = 50000 # size of the training set
n_test = 10000 # size of the test set
beta = 0.01 # value for regularization

activation_output = True # Activation function at the end
dropout_bool = False # Drop out
regularization = False # Regularization of the weights
weight_saver = True # Save the weights


#----Training set / Test set-----#
output_data = np.load('output_data.npy')
if translate:
    output_data = output_data + 2 # On only keep positive values for the ReLu at the end of the network !! TO BE ADAPTED !!
input_data = np.load('input_data.npy')

X_train, X_test, Y_train, Y_test = train_test_split(input_data, output_data, test_size=0.33, random_state=42)

x_train = X_train[:n_training,:]
y_train = Y_train[:n_training,:]
x_test = X_test[:n_test,:]
y_test = Y_test[: n_test,:]

N = x_train.shape[0]


#----Defining Tensors----#
with tf.variable_scope("place_holder"):
    x = tf.placeholder(tf.float32, (None, 144))
    y = tf.placeholder(tf.float32, (None, 2))
    keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("hidden_layer", reuse = tf.AUTO_REUSE):
    W_hidden = tf.get_variable("W_hidden",shape = [144, n_hidden], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    b_hidden = tf.get_variable("b_hidden",shape = [n_hidden,], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    x_hidden = tf.nn.relu(tf.matmul(x, W_hidden) + b_hidden) # ACTIVATION FUNCTION 
    
    if dropout_bool:
        x_hidden = tf.nn.dropout(x_hidden, keep_prob)

with tf.variable_scope("output", reuse = tf.AUTO_REUSE):
    W_output = tf.get_variable("W_output",shape = [n_hidden, 2], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    b_output = tf.get_variable("b_output",shape = [2,], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    output = tf.matmul(x_hidden, W_output) + b_output
    
    if activation_output:
        output = tf.nn.relu(tf.matmul(x_hidden, W_output) + b_output) # ACTIVATION FUNCTION 
    
with tf.variable_scope("loss", reuse = tf.AUTO_REUSE):
    l = tf.reduce_mean(tf.square(output - y))
    
    if regularization:
        l = tf.reduce_mean(l + beta * tf.nn.l2_loss(W_hidden) + beta * tf.nn.l2_loss(W_output) )
        
with tf.variable_scope("optim", reuse = tf.AUTO_REUSE):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)
    
if weight_saver:
    with tf.variable_scope("saver", reuse = tf.AUTO_REUSE):
        saver = tf.train.Saver()


#----Training Network----#
t1 = time.time()
loss_value = []

with tf.Session() as sess:
    step = 0
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        pos = 0
        while pos < N:
            batch_x = x_train[pos:pos + batch_size, :]
            batch_y = y_train[pos:pos + batch_size, :]
            if dropout_bool:
                feed_dict = {x:batch_x, y: batch_y, keep_prob:dropout_prob}
            else:
                feed_dict = {x:batch_x, y: batch_y}
            train, loss = sess.run([train_op,l], feed_dict=feed_dict)
            step += 1
            pos += batch_size
            loss_value.append(loss)
        print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
#         if epoch % 50 == 0:
#             y_test_train = sess.run(output, feed_dict = {x:x_train})
#             plt.scatter (y_test_train[:,0], y_test_train[:,1])
#             plt.show()
        
    t2 = time.time()
    print ('The training lasts %f' %(t2 - t1))
    
    
    #Save Weights
    if weight_saver:
        save_path = saver.save(sess,"tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)
    
    #Make predictions
    if dropout_bool:
        y_pred = sess.run(output, feed_dict = {x:x_test, keep_prob:1.0})
    else:
        y_pred = sess.run(output, feed_dict = {x:x_test})

#     print (y_pred)
#     print (y_test[0:10,:])
    


#----RMSE Error + Error Curve----#
with tf.Session() as sess:
    RMSE = tf.sqrt(tf.losses.mean_squared_error(y_pred, y_test))
    RMSE_value = sess.run(RMSE)
    mean_error_value =  mean_absolute_error(y_test, y_pred)
    print("RMSE error : {}".format(RMSE_value))
    print("Average of absolute differences : {}".format(mean_error_value))

# plt.scatter(np.arange(len(loss_value)), loss_value)
# plt.show()

