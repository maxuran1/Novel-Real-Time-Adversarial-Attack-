# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:18:43 2020

@author: owner
"""
import math 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

from label_wav import label_wav 
from label_wav import force_load
import input_data

import os

global_mfcc = 0

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  
  with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    
def test_graph(inputdata, graphfile):
    load_graph(graphfile)
    graph = tf.compat.v1.get_default_graph()
    output_tensor = graph.get_tensor_by_name('labels_softmax:0')
    init_mfcc = graph.get_tensor_by_name('Mfcc:0')
    input_tensor = graph.get_tensor_by_name('wav_data:0')
    
    with tf.compat.v1.Session() as sess:
        wav = inputdata.eval()
        mfcc, predictions = sess.run([init_mfcc, output_tensor], {input_tensor:wav})
        
    # print(predictions[0])
    # print(mfcc)
            
    return mfcc, predictions[0]

def load_wavs(directory):
    
    wavs = []
    
    for filename in os.listdir(directory):
        newwav = input_data.load_wav_file(os.path.join(directory, filename))
        wavs.append(newwav)
        
    return wavs

def attack_1():
    
    confusion_count = [0] * 12
    difference_vector = []
    delta_vector = []
    success_count = 0
    total_count = 0
    
    LEARNING_RATE = 0.5
    LABEL = [0] * 12
    LABEL[4] = 1
    CONSTANT = 1
    shape = (16000,)
    
    training = ** REPOSITORY NEEDED **
    test =  ** REPOSITORY NEEDED **
    graphfile = ** VICTIM GRAPH PATH NEEDED **
    
    trainset = load_wavs(training)
    testset = load_wavs(test)
    
    trainset = trainset[1:201]
    counter = 0
    
    with open("redo_attack1_up_75mask.txt", 'w') as f:

        for datum in trainset: 
            
            counter = counter + 1
            
            datum = audio_len(datum)
            init_ten = tf.convert_to_tensor(np.array(datum, dtype = np.float32))
            init_wav = tf.audio.encode_wav(tf.reshape(init_ten, (-1, 1)), 16000)
            raw_mfcc, raw_pred = test_graph(init_wav, graphfile)
            
            attack_vector, dede = adam_attack(datum, LABEL)
            
            predict, de = mfcc_to_prediction(raw_mfcc, attack_vector, 0.75)
            
            nppred = np.array(predict)
            
            print(nppred)
            
            print("Attack Number: " + str(counter))
            
            f.write("Attack Number: " + str(counter))  
            f.write("Raw Prediction " + str(raw_pred))
            f.write("Attack Predict " + str(predict))
            new_pred = np.argmax(nppred)
            
            if new_pred != 4:
                success_count = success_count + 1 
                
            total_count = total_count + 1 
            
            diff = raw_pred[4] - nppred[0][4]
            
            print(diff)
            confusion_count[new_pred] = confusion_count[new_pred] + 1
            difference_vector.append(diff)
            delta_vector.append(de)
        
        np_diff = np.array(difference_vector)
        np_delta = np.array(delta_vector)
        
        avg_diff = np.mean(np_diff)
        max_diff = np.max(np_diff)
        min_diff = np.min(np_diff)
        std_diff = np.std(np_diff)
        
        avg_delta = np.mean(np_delta)
        max_delta = np.max(np_delta)
        min_delta = np.min(np_delta)
        std_delta = np.std(np_delta)
            
        print("Total Number of Data Tested: " + str(total_count))
        print("Total Success: " + str(success_count))
        print("The std, min, average, max decrease in confidence are : " + str(std_diff) + ", " + str(min_diff) + "," + str(avg_diff) + "," + str(max_diff))
        print("The std, min, average, max delta are : " + str(std_delta) + ", " + str(min_delta) + "," + str(avg_delta) + "," + str(max_delta))
        print("The confusion matrix are: " + str(confusion_count))      
        
        f.write("Total Number of Data Tested: " + str(total_count))
        f.write("Total Success: " + str(success_count))
        f.write("The std, min, average, max decrease in confidence are : " + str(std_diff) + ", " + str(min_diff) + "," + str(avg_diff) + "," + str(max_diff))
        f.write("The std, min, average, max delta are : " + str(std_delta) + ", " + str(min_delta) + "," + str(avg_delta) + "," + str(max_delta))
        f.write("The confusion matrix are: " + str(confusion_count)) 
        f.write(str(np_diff))
        f.write(str(np_delta))

def attack_2():
    
    confusion_count = [0] * 12
    difference_vector = []
    delta_vector = []
    success_count = 0
    total_count = 0
    
    LEARNING_RATE = 0.5
    LABEL = [0] * 12
    LABEL[10] = 1
    CONSTANT = 1
    shape = (16000,)
    
    training = ** REPOSITORY NEEDED **
    test = ** REPOSITORY NEEDED **
    graphfile = ** VICTIM GRAPH NEEDED ** 
    
    trainset = load_wavs(training)
    testset = load_wavs(test)
    
    trainset = trainset[1:201]
    counter = 0
    
    with open("25partial_stop_log_200.txt", 'w') as f:

        for datum in trainset: 
            
            counter = counter + 1
            
            datum = audio_len(datum)
            init_ten = tf.convert_to_tensor(np.array(datum, dtype = np.float32))
            init_wav = tf.audio.encode_wav(tf.reshape(init_ten, (-1, 1)), 16000)
            raw_mfcc, raw_pred = test_graph(init_wav, graphfile)
            
            attack_vector, dede = adam_attack_partial(datum, LABEL)
            
            print(attack_vector)
            
            predict, de = mfcc_to_prediction(raw_mfcc, attack_vector, 0)
            
            nppred = np.array(predict)
            
            print("Attack Number: " + str(counter))
            
            print(raw_pred)
            print(nppred)
            
            f.write("Attack Number: " + str(counter))  
            f.write("Raw Prediction " + str(raw_pred))
            f.write("Attack Predict " + str(predict))
            new_pred = np.argmax(nppred)
            
            if new_pred != 10:
                success_count = success_count + 1 
                
            total_count = total_count + 1 
            
            diff = raw_pred[10] - nppred[0][10]
            
            print(diff)
            confusion_count[new_pred] = confusion_count[new_pred] + 1
            difference_vector.append(diff)
            delta_vector.append(de)
        
        np_diff = np.array(difference_vector)
        np_delta = np.array(delta_vector)
        
        avg_diff = np.mean(np_diff)
        max_diff = np.max(np_diff)
        min_diff = np.min(np_diff)
        std_diff = np.std(np_diff)
        
        avg_delta = np.mean(np_delta)
        max_delta = np.max(np_delta)
        min_delta = np.min(np_delta)
        std_delta = np.std(np_delta)
            
        print("Total Number of Data Tested: " + str(total_count))
        print("Total Success: " + str(success_count))
        print("The std, min, average, max decrease in confidence are : " + str(std_diff) + ", " + str(min_diff) + "," + str(avg_diff) + "," + str(max_diff))
        print("The std, min, average, max delta are : " + str(std_delta) + ", " + str(min_delta) + "," + str(avg_delta) + "," + str(max_delta))
        print("The confusion matrix are: " + str(confusion_count))      
        
        f.write("Total Number of Data Tested: " + str(total_count))
        f.write("Total Success: " + str(success_count))
        f.write("The std, min, average, max decrease in confidence are : " + str(std_diff) + ", " + str(min_diff) + "," + str(avg_diff) + "," + str(max_diff))
        f.write("The std, min, average, max delta are : " + str(std_delta) + ", " + str(min_delta) + "," + str(avg_delta) + "," + str(max_delta))
        f.write("The confusion matrix are: " + str(confusion_count)) 
        f.write(str(np_diff))
        f.write(str(np_delta))   
    
def attack_3():
    
    confusion_count = [0] * 12
    difference_vector = []
    delta_vector = []
    success_count = 0
    total_count = 0
    
    LEARNING_RATE = 0.5
    LABEL = [0] * 12
    LABEL[9] = 1
    CONSTANT = 1
    shape = (16000,)
    
    label = LABEL
    
    training = ** REPOSITORY NEEDED **
    test = ** REPOSITORY NEEDED **
    graphfile = ** VICTIM GRAPH NEEDED ** 
    
    bigset = load_wavs(training)
    
    testset = bigset[1:201]
    trainset = bigset[201:401]
    
    for element in range(len(trainset)):
        trainset[element] = audio_len(trainset[element])
    
    counter = 0
    
    with open("100_cross_off_log_200.txt", 'w') as f:
        
        attack_vector, dede = adam_attack_cross(trainset, label)

        for datum in testset: 
            
            counter = counter + 1
            
            datum = audio_len(datum)
            init_ten = tf.convert_to_tensor(np.array(datum, dtype = np.float32))
            init_wav = tf.audio.encode_wav(tf.reshape(init_ten, (-1, 1)), 16000)
            raw_mfcc, raw_pred = test_graph(init_wav, graphfile)
               
            predict, de = mfcc_to_prediction(raw_mfcc, attack_vector, 0)
            
            nppred = np.array(predict)
            
            print("Attack Number: " + str(counter))
            
            print(raw_pred)
            print(nppred)
            
            f.write("Attack Number: " + str(counter))  
            f.write("Raw Prediction " + str(raw_pred))
            f.write("Attack Predict " + str(predict))
            new_pred = np.argmax(nppred)
            
            if new_pred != 9:
                success_count = success_count + 1 
                
            total_count = total_count + 1 
            
            diff = raw_pred[9] - nppred[0][9]
            
            print(diff)
            confusion_count[new_pred] = confusion_count[new_pred] + 1
            difference_vector.append(diff)
            delta_vector.append(de)
        
        np_diff = np.array(difference_vector)
        np_delta = np.array(delta_vector)
        
        avg_diff = np.mean(np_diff)
        max_diff = np.max(np_diff)
        min_diff = np.min(np_diff)
        std_diff = np.std(np_diff)
        
        avg_delta = np.mean(np_delta)
        max_delta = np.max(np_delta)
        min_delta = np.min(np_delta)
        std_delta = np.std(np_delta)
            
        print("Total Number of Data Tested: " + str(total_count))
        print("Total Success: " + str(success_count))
        print("The std, min, average, max decrease in confidence are : " + str(std_diff) + ", " + str(min_diff) + "," + str(avg_diff) + "," + str(max_diff))
        print("The std, min, average, max delta are : " + str(std_delta) + ", " + str(min_delta) + "," + str(avg_delta) + "," + str(max_delta))
        print("The confusion matrix are: " + str(confusion_count))      
        
        f.write("Total Number of Data Tested: " + str(total_count))
        f.write("Total Success: " + str(success_count))
        f.write("The std, min, average, max decrease in confidence are : " + str(std_diff) + ", " + str(min_diff) + "," + str(avg_diff) + "," + str(max_diff))
        f.write("The std, min, average, max delta are : " + str(std_delta) + ", " + str(min_delta) + "," + str(avg_delta) + "," + str(max_delta))
        f.write("The confusion matrix are: " + str(confusion_count)) 
        f.write(str(np_diff))
        f.write(str(np_delta))       
        
def attack_4():
    
    confusion_count = [0] * 12
    difference_vector = []
    delta_vector = []
    success_count = 0
    total_count = 0
    
    LEARNING_RATE = 0.5
    shape = (16000,)
    
    smaller = 0
    smallerarray = []
    
    training = ** REPOSITORY NEEDED **
    test = ** REPOSITORY NEEDED **
    graphfile = ** VICTIM GRAPH NEEDED ** 
    
    bigset = load_wavs(training)
    
    testset = bigset[1:101]
    
    for element in range(len(testset)):
        true = 0
        if len(testset[element]) < 16000: 
            smaller = smaller + 1
            smallerarray.append(element)
            
        testset[element] = audio_len_backwards(testset[element], 0)
    
    counter = 0
    
    with open("off_predict_50mask1.txt", 'w') as f:

        for datum in testset: 
            
            counter = counter + 1
            
            print("Test Data Number: " + str(counter))
            
            init_ten = tf.convert_to_tensor(np.array(datum, dtype = np.float32))
            init_wav = tf.audio.encode_wav(tf.reshape(init_ten, (-1, 1)), 16000)
            raw_mfcc, raw_pred = test_graph(init_wav, graphfile)
           
            partial_datum = audio_len_backwards(datum, 0.5)
            if (counter - 1) == smallerarray[0]:
                print(datum)
                print(partial_datum)
            
            partial_ten = tf.convert_to_tensor(np.array(partial_datum, dtype = np.float32))
            partial_wav = tf.audio.encode_wav(tf.reshape(partial_ten, (-1, 1)), 16000)
            partial_mfcc, partial_pred = test_graph(partial_wav, graphfile)
               
            # print(raw_mfcc[0][30])
            # print(partial_mfcc[0][30])
            
            old_pred = np.argmax(raw_pred)
            new_pred = np.argmax(partial_pred)
            
            if new_pred == old_pred:
                print("Predict Success")
                success_count = success_count + 1 
                
            total_count = total_count + 1 
            
            confusion_count[new_pred] = confusion_count[new_pred] + 1


            
        print("Total Number of Data Tested: " + str(total_count))
        print("Total Success: " + str(success_count))
        print("The confusion matrix are: " + str(confusion_count))      
    
        f.write("Total Number of Data Tested: " + str(total_count))
        f.write("Total Success: " + str(success_count))
        f.write("The confusion matrix are: " + str(confusion_count))              
        
def mfcc_to_prediction(mfcc, attack_vector, perc_mask):
    graphfile = r'\tmp\my_frozen_graph.pb'

    index_limit = int(len(attack_vector[0]) * perc_mask)
        #Masking 
    for i in attack_vector:
        for time_slot_index in range(len(i)):
            if time_slot_index < index_limit:
                attack_vector[0][time_slot_index] = [0] * 40
                    
   #print(attack_vector)
                    
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        
        mfcc_input = tf.compat.v1.placeholder(tf.float32, shape = (1, 98, 40))
        attack_vec = tf.convert_to_tensor(attack_vector)
        
        mod_sqr = tf.math.square(attack_vec)
        delta = tf.compat.v1.reduce_sum(mod_sqr)
        
        attack = mfcc_input + attack_vec
        with tf.io.gfile.GFile(graphfile, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                import2 = tf.import_graph_def(graph_def, name='', input_map={"Mfcc:0":attack})
                
        output = graph.get_tensor_by_name("labels_softmax:0")
        
        with tf.compat.v1.Session() as sess:  
            out, delt = sess.run([output, delta], {mfcc_input: mfcc})
            
        return out, delt
            
             
def adam_attack(data, label): 

    LEARNING_RATE = 0.5
    label = [label]
    shape = (16000,)
    
    graphfile = ** VICTIM GRAPH FILE ** 
    global global_mfcc
       
    new_graph = tf.compat.v1.Graph()
    with new_graph.as_default():
    
        #load new tensors
    

        data_in = tf.compat.v1.placeholder(tf.float32, shape=(16000,))
        data_reshap = tf.reshape(data_in, (16000, 1))
        truth_label = tf.compat.v1.placeholder(tf.float32, shape=(1,12))
        mfc_mod = tf.Variable(np.zeros((1, 98, 40), dtype=np.float32))

        #Load subgraph to produce mfcc 
        with tf.io.gfile.GFile(graphfile, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            pruned_graph = tf.compat.v1.graph_util.extract_sub_graph(graph_def, ["AudioSpectrogram", "Mfcc/sample_rate", "Mfcc"])
            import1 = tf.import_graph_def(pruned_graph, name='', input_map={"decoded_sample_data":data_reshap})
            
        pre_mfcc = new_graph.get_tensor_by_name('Mfcc:0')
        attack = pre_mfcc + mfc_mod
        
        with tf.io.gfile.GFile(graphfile, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            import2 = tf.import_graph_def(graph_def, name='', input_map={"Mfcc:0":attack})
            
        #print(new_graph.get_operations())
  
        #load existing tensors
        victim_output = new_graph.get_tensor_by_name('labels_softmax:0')
        
        #Does not actually connect to graph, but create new node to nowhere.
        #rando2 = sample_data + rando

        #print(victim_output)
        
        reshape_op = new_graph.get_operation_by_name("Reshape_3")      
        # print(reshape_op)
        
        #calculate delta
        mod_sqr = tf.math.square(mfc_mod)
        delta = tf.compat.v1.reduce_sum(mod_sqr)
        
        #calculate loss 
        max_arg = tf.math.argmax(truth_label[0])
        
        # #Mean Squared Error
        #difference = tf.compat.v1.losses.mean_squared_error(truth_label, victim_output)
        #mod_loss = -1 * difference
        
        # #Inverse Max Element 
        difference = truth_label[0][max_arg] - victim_output[0][max_arg]
        print(truth_label)
        print(victim_output)
        mod_loss = 1/difference 
        
        total_loss = delta + 100 * mod_loss
        
        with open("new_graph_ops.txt", "w+") as file:
              for op in new_graph.get_operations():
                  file.write(str(op))

        #setup optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE)
        train = optimizer.minimize(total_loss, var_list=mfc_mod)
        
        
        init_op = tf.compat.v1.global_variables_initializer()


        with tf.compat.v1.Session() as sess2:  
            sess2.run([init_op])
            for x in range(10):
                pre_mfc, vec, att, loss, softmax, d, ml, _ = sess2.run([pre_mfcc, mfc_mod, attack, total_loss, victim_output, delta, difference, train], {data_in: data, truth_label: label})
                #opt = modifier.eval()
                # print("Training set: " + str(x))
                # print("pre_mfcc value is " + str(pre_mfc))
                # print("mfcc vector value is " + str(vec))
                # print("The Softmax is " + str(softmax))
                # print("The attacking array is" + str(att))
                # print("Delta is " + str(d))
                # print("Difference of confidence is " + str(ml))
                # print("Total loss is " + str(loss) + '\n\n')
                
                global_mfcc = pre_mfc
                
                
            #last_loss = total_loss.eval()
            
       #print(global_mfcc)
        return(vec, d)
          
def adam_attack_partial(data, label): 

    LEARNING_RATE = 0.5
    label = [label]
    shape = (16000,)
    
    graphfile = ** VICTIM GRAPH FILE ** 
    global global_mfcc
       
    new_graph = tf.compat.v1.Graph()
    with new_graph.as_default():
    
        #load new tensors
    
        added_vec = tf.constant(np.zeros((1, 73, 40), dtype = np.float32))
        data_in = tf.compat.v1.placeholder(tf.float32, shape=(16000,))
        data_reshap = tf.reshape(data_in, (16000, 1))
        truth_label = tf.compat.v1.placeholder(tf.float32, shape=(1,12))
        mfc_mod = tf.Variable(np.zeros((1, 25, 40), dtype=np.float32))

        #Load subgraph to produce mfcc 
        with tf.io.gfile.GFile(graphfile, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            pruned_graph = tf.compat.v1.graph_util.extract_sub_graph(graph_def, ["AudioSpectrogram", "Mfcc/sample_rate", "Mfcc"])
            import1 = tf.import_graph_def(pruned_graph, name='', input_map={"decoded_sample_data":data_reshap})
            
        pre_mfcc = new_graph.get_tensor_by_name('Mfcc:0')
        intermediate = tf.concat([added_vec, mfc_mod], 1)
        attack = pre_mfcc + intermediate
        
        with tf.io.gfile.GFile(graphfile, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            import2 = tf.import_graph_def(graph_def, name='', input_map={"Mfcc:0":attack})
            
        #print(new_graph.get_operations())
  
        #load existing tensors
        victim_output = new_graph.get_tensor_by_name('labels_softmax:0')
        
        #Does not actually connect to graph, but create new node to nowhere.
        #rando2 = sample_data + rando

        #print(victim_output)
        
        reshape_op = new_graph.get_operation_by_name("Reshape_3")      
        # print(reshape_op)
        
        #calculate delta
        mod_sqr = tf.math.square(mfc_mod)
        delta = tf.compat.v1.reduce_sum(mod_sqr)
        
        #calculate loss 
        max_arg = tf.math.argmax(truth_label[0])
        
        # #Mean Squared Error
        #difference = tf.compat.v1.losses.mean_squared_error(truth_label, victim_output)
        #mod_loss = -1 * difference
        
        # #Inverse Max Element 
        difference = truth_label[0][max_arg] - victim_output[0][max_arg]
        mod_loss = 1/difference 
        
        total_loss = delta + 100 * mod_loss
        
        with open("new_graph_ops.txt", "w+") as file:
              for op in new_graph.get_operations():
                  file.write(str(op))

        #setup optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE)
        train = optimizer.minimize(total_loss, var_list=mfc_mod)
        
        
        init_op = tf.compat.v1.global_variables_initializer()


        with tf.compat.v1.Session() as sess2:  
            sess2.run([init_op])
            for x in range(10):
                pre_mfc, vec, att, loss, softmax, d, ml, _ = sess2.run([pre_mfcc, intermediate, attack, total_loss, victim_output, delta, difference, train], {data_in: data, truth_label: label})
                #opt = modifier.eval()
                # print("Training set: " + str(x))
                # print("pre_mfcc value is " + str(pre_mfc))
                # print("mfcc vector value is " + str(vec))
                # print("The Softmax is " + str(softmax))
                # print("The attacking array is" + str(att))
                # print("Delta is " + str(d))
                # print("Difference of confidence is " + str(ml))
                # print("Total loss is " + str(loss) + '\n\n')
                
                global_mfcc = pre_mfc
                
                
            #last_loss = total_loss.eval()
            
       #print(global_mfcc)
        return(vec, d)    
        
def adam_attack_cross(dataset, label): 

    LEARNING_RATE = 0.01
    label = [label]
    shape = (16000,)
    
    graphfile = ** VICTIM GRAPH FILE ** 
    global global_mfcc
       
    new_graph = tf.compat.v1.Graph()
    with new_graph.as_default():
    
        #load new tensors
    
        #added_vec = tf.constant(np.zeros((1, 73, 40), dtype = np.float32))
        data_in = tf.compat.v1.placeholder(tf.float32, shape=(16000,))
        data_reshap = tf.reshape(data_in, (16000, 1))
        truth_label = tf.compat.v1.placeholder(tf.float32, shape=(1,12))
        mfc_mod = tf.Variable(np.zeros((1, 98, 40), dtype=np.float32))

        #Load subgraph to produce mfcc 
        with tf.io.gfile.GFile(graphfile, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            pruned_graph = tf.compat.v1.graph_util.extract_sub_graph(graph_def, ["AudioSpectrogram", "Mfcc/sample_rate", "Mfcc"])
            import1 = tf.import_graph_def(pruned_graph, name='', input_map={"decoded_sample_data":data_reshap})
            
        pre_mfcc = new_graph.get_tensor_by_name('Mfcc:0')
        #intermediate = tf.concat([added_vec, mfc_mod], 1)
        #attack = pre_mfcc + intermediate
        
        attack= pre_mfcc + mfc_mod
        
        with tf.io.gfile.GFile(graphfile, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            import2 = tf.import_graph_def(graph_def, name='', input_map={"Mfcc:0":attack})
            
        #print(new_graph.get_operations())
  
        #load existing tensors
        victim_output = new_graph.get_tensor_by_name('labels_softmax:0')
        
        #Does not actually connect to graph, but create new node to nowhere.
        #rando2 = sample_data + rando

        #print(victim_output)
        
        reshape_op = new_graph.get_operation_by_name("Reshape_3")      
        # print(reshape_op)
        
        #calculate delta
        mod_sqr = tf.math.square(mfc_mod)
        delta = tf.compat.v1.reduce_sum(mod_sqr)
        
        #calculate loss 
        max_arg = tf.math.argmax(truth_label[0])
        
        # #Mean Squared Error
        #difference = tf.compat.v1.losses.mean_squared_error(truth_label, victim_output)
        #mod_loss = -1 * difference
        
        # #Inverse Max Element 
        difference = truth_label[0][max_arg] - victim_output[0][max_arg]
        mod_loss = 1/difference 
        
        total_loss = delta + 1000 * mod_loss
        
        with open("new_graph_ops.txt", "w+") as file:
              for op in new_graph.get_operations():
                  file.write(str(op))

        #setup optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE)
        train = optimizer.minimize(total_loss, var_list=mfc_mod)
        
        
        init_op = tf.compat.v1.global_variables_initializer()


        with tf.compat.v1.Session() as sess2:  
            sess2.run([init_op])
            for x in range(10):
                count = 0
                for datum1 in dataset:
                    count = count + 1 
                    pre_mfc, vec, att, loss, softmax, d, ml, _ = sess2.run([pre_mfcc, mfc_mod, attack, total_loss, victim_output, delta, difference, train], {data_in: datum1, truth_label: label})
                #opt = modifier.eval()
                # print("Training set: " + str(x))
                # print("pre_mfcc value is " + str(pre_mfc))
                # print("mfcc vector value is " + str(vec))
                # print("The Softmax is " + str(softmax))
                # print("The attacking array is" + str(att))
                # print("Delta is " + str(d))
                # print("Difference of confidence is " + str(ml))
                # print("Total loss is " + str(loss) + '\n\n')
                    print("This is step " + str(x) + " with count " + str(count) + " and the loss is: " + str(loss))
                
                global_mfcc = pre_mfc
                
                
            #last_loss = total_loss.eval()
            
       #print(global_mfcc)
        return(vec, d)    
    
def audio_len(audio):
    #Takes any audio input array and makes it 16000 samples (add if less, subtract if more)
    if len(audio) < 16000:
        difference = 16000 - len(audio)
        audio_eq = [0] * difference
        
        success = audio_eq.extend(audio)
        
    
    elif len(audio) > 16000:
        audio_eq = audio[0:16000]
    else:
        audio_eq = audio
        
    if (len(audio_eq) != 16000):
        raise ValueError ("audio_len not working properly")
        
    return audio_eq
    
def audio_len_backwards(audio, mask):
    #Takes any audio input array and makes it 16000 samples (add if less, subtract if more)
    show = 1-mask 
    show_index = int(show * 16000)
    mask_index = 16000 - int(show * 16000)
    
    audio = list(audio)
    
    mask_array = [0] * mask_index 
    
    if len(audio) < 16000:
        difference = 16000 - len(audio)
        audio_eq = [0] * difference
        
        success = audio_eq.extend(audio)
        
        audio = audio_eq
    
    elif len(audio) > 16000:
        audio = audio[0:16000]
    else:
        audio = audio
        
    if (len(audio) != 16000):
        raise ValueError ("audio_len not working properly")
        
    if mask != 0:
        short = audio[0:show_index]
        succ = short.extend(mask_array)
    
        return short
    else:
        return audio

def keras_loss_function(data_actual, data_predicted):
    
    datasetnp = data_predicted.numpy()
    dataset = datasetnp.tolist()
    
    
    CONSTANT = 1
    #Find sqrt(d|^2)
    delta_squ = 0
    desire = [0] * 12
    desire[10] = 1
    
    for element in dataset:
        delta_squ = delta_squ + element ** 2 
    
    delta_squ = math.sqrt(delta_squ)
    
    #Generate wav file from data
    wav = gen_wav(dataset)
    att_loss = loss_function(wav, desire)
    
    return delta_squ + CONSTANT * att_loss
    
def loss_function(wavfile, trulabel, pass_tensor = False): 
    
    #wav = r"C:\Users\owner\Documents\weirdoff.wav"
    wav = wavfile
    labels = ** VICTIM LABEL FILE ** 
    graph = ** VICTIM GRAPH FILE ** 
    input_name = 'wav_data:0'
    output_name = 'labels_softmax:0'
    how_many_labels = 12
    
    #result = label_wav(wav, labels, graph, input_name, output_name, how_many_labels, pass_tensor)
    result = force_load(wav, labels, graph, input_name, output_name, how_many_labels, pass_tensor)
    sort_result = sorted(result, key=lambda x: x[0])
    
    scores = []
    for work in sort_result:
        scores.append(work[2])
        
    desired = trulabel
    #loss = mean_squared(scores, desired)
    #loss = max_distance_of_top(scores, desired)
    loss = inverse_max_d_of_top(scores, desired)
    #print(scores)
    #print(loss)
    
    return loss

    
    
def mean_squared (score, desire):
    #calcualte the mean squared distance between two vectors, they must be of equal length
    
    if len(score) != len(desire):
        raise ValueError("Lengths of vectors not the same")
    
    sum = 0
    for element in range(len(score)):
        sum = sum + (desire[element] - score[element])**2
    
    return math.sqrt(sum)

def max_distance_of_top (score, desire):
    if len(score) != len(desire):
        raise ValueError("Lengths of vectors not the same")
    
    maxa = 0
    count = 0
    index = -1
    for element in desire:
        if element > maxa:
            maxa = element
            index = count
        count = count + 1 
        
    print(index)
    
    if index == -1:
        raise ValueError ("Invalid values in desire")
        
    return desire[index] - score[index]
            
def inverse_max_d_of_top (score, desire):
    
    max_to_top = max_distance_of_top(score, desire)
    print(max_to_top)
    
    if max_to_top != 0:
        return 1/max_to_top
    else:
        return float("inf")
    

def FGSM_mod():
    
    BLOCK_WIDTH = 20
    
    soundfile = ** SOUND FILE ** 

    sound = input_data.load_wav_file(soundfile)
        
    newsound = sound
    tempsound = sound
    
    attack = []
    
    #[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    

    
    for element in range(0, len(sound), BLOCK_WIDTH):
        current = np.array(tempsound[element:element+BLOCK_WIDTH])
        abs_current = np.abs(current)
        average = np.average(abs_current)
        curr_sound = tempsound[element:element+BLOCK_WIDTH]
        
        print(element)
        
        if average > 0.01:
            
            vector1 = [0, 0, 0.2, 0.2, 0.2, 0.1, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0.2, 0.1, 0, 0, 0.1]
            vector2 = [0, 0, -0.2, -0.2, -0.2, -0.1, 0, 0, 0 , 0, 0, 0, 0, 0, 0, -0.2, -0.1, 0, 0, -0.1]
            
            # for x in range(BLOCK_WIDTH):
            #     if (x % 2 == 0):
            #         vector1.append(0.2)
            #     else:
            #         vector1.append(0)
            #     if (x % 5 == 0):
            #         vector2.append(0.2)
            #     else:
            #         vector2.append(0)
                    
            high_sound = list_adder(curr_sound, vector1)
            low_sound = list_adder(curr_sound, vector2)
            
            bsline = gen_wav(tempsound)
            baseline_loss = loss_function(bsline, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            
            tempsound[element:element+BLOCK_WIDTH] = high_sound
            wav = gen_wav(tempsound)
            #input_data.save_wav_file(r"C:\Users\owner\Documents\onedit.wav", newsound, 16000)
            high_loss = loss_function(wav, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
                
            tempsound[element:element+BLOCK_WIDTH] = low_sound
            #input_data.save_wav_file(r"C:\Users\owner\Documents\onedit.wav", newsound, 16000)
            wav = gen_wav(tempsound)
            low_loss = loss_function(wav, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            
            print(high_loss)
            print(low_loss)
            print(baseline_loss)
            
            if (high_loss < low_loss):
                if (high_loss < baseline_loss):
                    newsound[element:element+BLOCK_WIDTH] = high_sound
                    for e in vector1:
                        attack.append(e)
                    print("added high")
                else:
                    newsound[element:element+BLOCK_WIDTH] = curr_sound
                    print("same")
                    for e in vector1:
                        attack.append(0)
            else:
                if (low_loss < baseline_loss):
                    newsound[element:element+BLOCK_WIDTH] = low_sound
                    print("added low")
                    for e in vector2:
                        attack.append(e)
                else:
                    newsound[element:element+BLOCK_WIDTH] = curr_sound
                    print("same")
                    for e in vector1:
                        attack.append(0)
                    
            tempsound = sound 
            
        else:
            for e in range(BLOCK_WIDTH):
                attack.append(0)
                
            
            #tempsound[element] = tempsound[element] - 0.1
            #wav = gen_wav(tempsound)
            #downloss = loss_function(wav, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            
            #tempsound[element] = tempsound[element] + 0.2
            #wav = gen_wav(tempsound)
            #uploss = loss_function(wav, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            
            #tempsound[element] = tempsound[element] - 0.1
            
            # tempsound[element] = tempsound[element] - 0.1
            # input_data.save_wav_file(r"C:\Users\owner\Documents\onedit.wav", tempsound, 16000)
            # downloss = loss_function(r"C:\Users\owner\Documents\onedit.wav", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            # tempsound[element] = tempsound[element] + 0.2
            # input_data.save_wav_file(r"C:\Users\owner\Documents\onedit.wav", newsound, 16000)
            # uploss = loss_function(r"C:\Users\owner\Documents\onedit.wav", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            # tempsound[element] = tempsound[element] - 0.1
            
            # if downloss > uploss:
            #     newsound.append(sound[element]-0.1)
            # else:
            #     newsound.append(sound[element]+0.1)
        
        #print(element)
            
    print(len(attack))
    
def list_adder(list1, list2):
    if (len(list1) != len(list2)):
        raise ValueError ("Lengths of Lists not Equal")
        
    newlist = []
    for i in range(len(list1)):
        newlist.append(list1[i] + list2[i])
        
    return newlist
        
    
def gen_wav(sound):
    
    wave = np.reshape(sound, (-1, 1))
    file = tf.audio.encode_wav(wave, 16000) 
    with tf.compat.v1.Session() as sess:  wavfile = (file.eval())
    
    return wavfile


    