#!/usr/bin/env python
# coding: utf-8

# This function return all of the data which have been pre-processed before training the neural network

#----Import Packages----#
from struct import *
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from parameters_ML_optical_flow import *



#----Parameters----#
args = parser.parse_args()

input_file = args.input
output_file = args.output_speed

sz = (args.height, args.width)
img = np.zeros(sz)

tau = 5000 # Normalization Value for the exponential decay
delta_t = 1000 # Value to remove old events



#----Useful Functions----#
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


#----Input----#
print('Input being prepared with Corner Detection')

time_stamp = []
input_data = []
with open(input_file, 'r') as events:
    vec_decrease = np.vectorize(decrease)
    
    for ev in events:
        [ts, x, y, p] = map(int, ev.split())
        img[x,y] = ts
        patch = img[(x-6):(x+6), (y-6):(y+6)]
        
        if ts % 1000000 == 0:
            print(ts/7510000*100)
            
        if patch.shape == (12, 12):
            patch_update = vec_decrease(patch, ts)
            if is_corner_threshold(patch_update):
                input_data.append(patch_update.reshape(144,1))
                time_stamp.append(ts)
#                 if ts > 500000:
#                     print(patch_update)
#                     plt.imshow(patch_update, cmap="gray", vmin=0, vmax=1)
#                     plt.show()



#----Output----#
print('Output being prepared')
output = []
i = 0
with open(output_file, 'r') as events:
    for ev in events:
        ev = ev.split()
        for i in range(1000):
            output.append([float(ev[3]), float(ev[4])])


#Only keep the right output (corner)
print('Output to be kept')
output_data = []

for i in range(len(output)):
    if len(time_stamp) == 0:
        break
    elif i == time_stamp[0]:
        output_data.append(output[i])
        del time_stamp[0]


#----Save Data----#
output_data = np.asarray (output_data)
input_data = np.asarray (input_data)
input_data = np.squeeze(input_data, axis=2)

np.save('output_data.npy', output_data)
np.save('input_data.npy', input_data)

#---- Quick overview
#for i in range(20):
#    plt.imshow(input_data[i].reshape(12,12), cmap="gray", vmin = 0, vmax =1)
#    plt.show()
#    
#print(output_data.shape)
