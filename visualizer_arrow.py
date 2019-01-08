#!/usr/bin/python

from struct import *
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pylab

ev_file = os.path.abspath(".") + "/input/Bar_24x2_directions.txt"

###################
### PARAMETERS ####
###################


pylab.ion()
t_norm = 10
tau = 500
delta_t = 1000
sz = (304, 240)
img = 0.5 * np.ones(sz)
img_to_process = np.zeros(sz)

step = 300
loc = 0


n_hidden = 60
activation_output = False

#########################
### Useful Functions ####
#########################

def decrease (t_old, t_current):
    if t_old == 0:
        val_decrease = 0.
    else:
        val_decrease = round(np.exp(-(t_current - t_old)/tau),3)
    return val_decrease

def is_corner_threshold(patch_update):
    sum_down = 0
    sum_up = 0
    sum_right = 0
    sum_left = 0
    
    for i in range(11):
        if patch_update[i,0] > 0.4:
            sum_up += patch_update[i,0]
            
        if patch_update[i,11] > 0.4:
            sum_down += patch_update[i,11]
            
        if patch_update[0,i] > 0.4:
            sum_left += patch_update[0,i]    
        
        if patch_update[11,i] > 0.4:
            sum_right += patch_update[11,i]
    if (sum_down < 1.4 and sum_right < 1.4) or (sum_down < 1.4 and sum_left < 1.4) or (sum_up < 1.4 and sum_right < 1.4) or (sum_up < 1.4 and sum_left < 1.4):
        return True
    else:
        return False

######################
### IMPORT WEIGHTS ###
######################

tf.reset_default_graph()


W_hidden = tf.get_variable("hidden_layer/W_hidden", shape=[144,n_hidden])
b_hidden = tf.get_variable("hidden_layer/b_hidden", shape=[n_hidden,])
W_output = tf.get_variable("output/W_output", shape=[n_hidden,2])
b_output = tf.get_variable("output/b_output", shape=[2,])

with tf.variable_scope("place_holder"):
    x_input = tf.placeholder(tf.float32, (1, 144))
    y_input = tf.placeholder(tf.float32, (1, 2))
    
with tf.variable_scope("hidden_layer", reuse = tf.AUTO_REUSE):
    x_hidden = tf.nn.relu(tf.matmul(x_input, W_hidden) + b_hidden)
    
with tf.variable_scope("output", reuse = tf.AUTO_REUSE):

	if activation_output:
		output = tf.nn.relu(tf.matmul(x_hidden, W_output) + b_output)
	else :
		output = tf.matmul(x_hidden, W_output) + b_output


saver = tf.train.Saver()

####################
### SHOW FIGURE ###
###################


f = plt.figure()
with tf.Session() as sess:
	tensor_path = os.path.abspath(".") + "/tmp/model.ckpt"
	saver.restore(sess, tensor_path)

	with open(ev_file, "r") as events:

		vec_decrease = np.vectorize(decrease)
		for ev in events:
			ev = map(int, ev.split())
			[ts, x, y, p] = ev
			ts = ts/t_norm
			img[x,y] = p
			img_to_process[x,y] = ts
			patch = img_to_process[(x-6):(x+6), (y-6):(y+6)]
	
			if patch.shape == (12,12):
				patch_update = vec_decrease(patch, ts)
				if is_corner_threshold (patch_update):	
					speed = sess.run(output, feed_dict = {x_input:patch_update.reshape(1,144)})
					[vx, vy] = 15*(speed[0,] - 2)
					plt.arrow(y, x, vy, vx, color='b', width= 1, head_width = 2)

			if ts > step*loc:
				plt.imshow(img, cmap="gray", vmin=0, vmax=1)
				loc += 1

				plt.show()
				pylab.draw()
				img = 0.5*np.ones(sz)
				f.canvas.flush_events()
				f.clf()

