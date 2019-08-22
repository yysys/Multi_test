import os, sys
import pickle

import tensorflow as tf
from tensorflow.python.client import device_lib

def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def pickle_load(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def average_tower_grads(tower_grads):  
    print('towerGrads:')  
    idx = 0  
    for grads in tower_grads:  # grads 为 一个list，其中元素为 梯度-变量 组成的二元tuple  
        print('grads---tower_%d' % idx)  
        for g_var in grads:  
            print('\t%s\n\t%s' % (g_var[0].op.name, g_var[1].op.name))  
        idx += 1  
      
    if(len(tower_grads) == 1):  
        return tower_grads[0]  
    avgGrad_var_s = []  
    for grad_var_s in zip(*tower_grads):  
        grads = []  
        v = None  
        for g, v_ in grad_var_s:  
            g = tf.expand_dims(g, axis=0)  
            grads.append(g)  
            v = v_  
        all_g = tf.concat(grads, axis=0)  
        avg_g = tf.reduce_mean(all_g, 0, keepdims=False)  
        avgGrad_var_s.append((avg_g, v));  
    return avgGrad_var_s 