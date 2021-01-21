# Coding : UTF-8
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import time
import json
import pickle
import string
from string import digits
import re
import os

from helpers import load_config, Max_length


"""
Load the Data
Data format should be parallel .txt files for file, src, trg
* file file(.files) should contain the name of each sequence on a new line
* src file(.gloss) should contain a new source input of each sentence on each line
* trg file(.skels) should contain skeleton data, with each line a new sequence, each frame following on from the previous
Each joint value should be separated by a space; " "
Each frame is partioned using the known trg_size length, which includes 2D joints
"""


################################################

def make_y(key, trg_size):
    """
    trg data(keypoints) have a 254 pose keypoints and Count Value(CE)
    using CE in train and reshape to Tacotron train model 
    -> (number of sentence, max frame number, trg_size)

    :param array key: keypoints data
    :param int trg_size: vocab_size_target, each frame keypoints count
    """
    
    tmp = []

    for i in range(len(key)):
        frame = key[i]
        tmp_list = []

        cnt = int(len(frame.split(' '))/trg_size)

        for j in range(cnt): 
            temp = []
            key_frame = frame.split(' ')[trg_size*j:trg_size*(j+1)] # contains CE
            for k in range(trg_size): # trg_size number of keypoints
                temp.append(float(key_frame[k]))

            tmp_list.append(str(temp).replace('[','').replace(']',''))

        tmp.append(tmp_list)
    
    y_data = pd.DataFrame(tmp)
    y_data = y_data.transpose()

    y=[]

    for i in range(y_data.shape[1]):
        frame= y_data[i].tolist()
        tmp=[]

        for j in range(len(frame)):
            tmp_list=[]

            if frame[j] == None:
                tmp_list=[0 for _ in range(trg_size)]

            else:
                key_frame = frame[j].split(",")
                for k in range(trg_size):
                    tmp_list.append(float(key_frame[k]))

            tmp.append(tmp_list)
        y.append(tmp)
    
    y=np.array(y)

    return y

################################################

def make_output_data(filename, mor, key, trg_size):
    """
    Preprocessing files to make json, image, video

    :param array filename: files data(.files)
    :param array mor: src data(.gloss)
    :param array key: trg data(.skels)
    :param int trg_size: vocab_size_target, each frame keypoints count
    """
    
    output_file = [file.split('SL_')[1].split('_F')[0] for file in filename]
    output_gloss = [glo.replace(' ', '_') for glo in mor]
    output_skels = [len(k.split(' '))//trg_size for k in key]
    
    return output_file, output_gloss, output_skels

################################################

# Main load data func
def load_txt_data(path, files, src, trg):   
    """
    Load files, src, trg data

    - param
    :param string files: .files name
    :param string src: .gloss name
    :param string trg: .skels name
    - return
      array length -> (number of sentence)
    :param array file: 'filename'
    :param array mor: 'I, like, an, apple'
    :param array key: '0.0122 0.13344 .....'
    """
    
    # Load file name 
    file = []
    with open(path + '.' + files, "r", encoding="UTF-8", errors='ignore') as f:
        f = f.readlines() 
        for i in range(len(f)):
            f[i] = f[i].replace('\n','')
            file.append(f[i])

    # Load morpheme gloss
    mor = []
    with open(path + '.' + src, "r", encoding="UTF-8", errors='ignore') as f:
        f = f.readlines() 
        for i in range(len(f)):
            f[i] = f[i].replace('\n','')
            mor.append(f[i])

    # Load keypoint skels
    with open(path + '.' + trg, 'r') as f:
        key = f.readlines()
        f.close()

    return file, mor, key

################################################

def load_data(cfg:dict, mode):
    """
    Load train, dev, test data
    And then data filtered to include sentences up to `max_sent_length`
    on source and target side
    
    :param cfg data_cfg: configuration dictionary for data ("data" part of configuation file)
    - return
    :param array X_:
    :param array y:
    :param array decoder_input_array:
    :param array mel_spectro_data_array:
    + if train
    :param int max_X: Max Vocabulary number in sentence
    :param int vocab_size_source: number of Word Embedding 
    """
    
    # Load cfg =========================================
    data_cfg = cfg["data"]
    
    # Source, Target and Files postfixes
    src = data_cfg["src"] # gloss
    trg = data_cfg["trg"] # skels
    files = data_cfg.get("files", "files") # files
    
    # Train, Dev and Test Path
    path = data_cfg[mode]
    data_path = cfg["data_path"]
    
    max_sent_length = data_cfg["max_sent_length"]
    
    model_cfg = cfg["model"]
    vocab_size_target = model_cfg["vocab_size_target"]
    
    
    # Load data =========================================
    file, mor, key = load_txt_data(path, files, src, trg)
    
    
    # make output_files =========================================
    """
    Choose the 1) or 2)
    1) If you run this code First time = file does not exist
    2) If you already run this code, so file is exist
    """
    
    # 1) First time, file does not exist ==================
    out_files, out_gloss, out_skels = make_output_data(file, mor, key, vocab_size_target)    
    
    # Save preprocessing data
    with open(data_path+'out_files_'+str(mode)+'.pickle', 'wb') as f:
        pickle.dump(out_files, f)    
    with open(data_path+'out_gloss_'+str(mode)+'.pickle', 'wb') as f:
        pickle.dump(out_gloss, f)   
    with open(data_path+'out_skels_'+str(mode)+'.pickle', 'wb') as f:
        pickle.dump(out_skels, f)    
    
   
    # 2) Second time, file is exist ==================
    # # Load preprocessing data
    # with open(data_path+'out_files_'+str(mode)+'.pickle', 'rb') as f:
    #     out_files = pickle.load(f)

    # with open(data_path+'out_gloss_'+str(mode)+'.pickle', 'rb') as f:
    #     out_gloss = pickle.load(f)

    # with open(data_path+'out_skels_'+str(mode)+'.pickle', 'rb') as f:
    #     out_skels = pickle.load(f)
    
    
    # Make X and Y =========================================
    """
    Choose the 1) or 2)
    1) If you run this code First time = file does not exist
    2) If you already run this code, so file is exist
    """
    
    # 1) First time, file does not exist ==================
    # Make X
    X = pd.DataFrame(mor)[0].values
    # Make y
    y = make_y(key, vocab_size_target)
    y = np.array(y)
    
    # Save preprocessing file
    np.save(data_path+'X_'+str(mode)+'.npy',X)
    
    with open(data_path+'y_'+str(mode)+'.pickle', 'wb') as f:
        pickle.dump(y, f)

    # 2) Second time, file is exist ==================
    # # Load preprocessing data
    # X = np.load(data_path+'X_'+str(mode)+'.npy', allow_pickle=True)

    # with open(data_path+'y_'+str(mode)+'.pickle', 'rb') as f:
    #     y = pickle.load(f)
    # y = np.array(y)
    
    
    # Embedding =========================================
    
    # Load trial.gloss: is contains train, test, dev
    mor_trial = []
    with open(data_path + 'trial.gloss', "r", encoding="UTF-8", errors='ignore') as f:
        f = f.readlines() 
        for i in range(len(f)):
            f[i] = f[i].replace('\n','')
            mor_trial.append(f[i])
    
    X_trial = pd.DataFrame(mor_trial)[0].values

    # max_X : Max Vocabulary number in sentence
    max_X= Max_length(X_trial)
    max_y= vocab_size_target
    
    Tok = Tokenizer()
    Tok.fit_on_texts(X_trial)

    word2index = Tok.word_index
    # vocab_size_source : number of Word Embedding  
    vocab_size_source = len(word2index) + 1

    X_ = Tok.texts_to_sequences(X)
    # Padding sentence
    X_ = pad_sequences(X_, maxlen=max_X, padding='post')
    X_ = np.array(X_)
    
    
    # Make Model input Values =========================================
    
    # t: frame number of each sentence
    t = out_skels[0]
    mt = max(out_skels)
    kk = max_sent_length-mt
    
    # Padding y (?,frame number,?) -> (?,Max frame number,?)
    y = tf.concat((y[:,:,:], tf.zeros([y.shape[0], kk, vocab_size_target], dtype=tf.float64)), 1).numpy()
    
    decod_inp = tf.concat((y[0,:t-1, :], tf.zeros_like(y[0, t-1:, :])), 0)
    decod_inp = tf.reshape(decod_inp, [1, max_sent_length, vocab_size_target])
    
    for i in range(1,len(out_skels)):
        t = out_skels[i]
        de = tf.concat((y[i,:t-1, :], tf.zeros_like(y[i, t-1:, :])), 0)
        de = tf.reshape(de, [1, max_sent_length, vocab_size_target])
        decod_inp = tf.concat((decod_inp, de), 0)
    
    # Make decoder input array
    decoder_input_array = decod_inp[:]
    
    mel_spectro_data_array = y[:]
    
    
    # Return is different then train or not train
    if mode == "train":
        return X_, y, decoder_input_array, mel_spectro_data_array, max_X, vocab_size_source
    elif mode != "train":
        return X_, y, decoder_input_array, mel_spectro_data_array