import numpy as np
import cv2
import json
import os

from model import build_model
from key2video import create_img_video
from dtw import calculate_dtw
from helpers import make_dir


def make_predict(cfg:dict, model, X_, y, decoder_input_array,
                 output_file, output_gloss, output_skels,
                 result_path, epoch=None, best=False):
    """
    Make a predict and a keypoints json file
    
    :param tf.model model: load training model
    :param string result_path: path to save keypoints json and img/video
    :param array output_skels: each sentence's frame count
    :param int epoch: train epoch (default None) 
    :param T/F best: best Y/N
    """
    
    # save_path / img_path : path to save keypoints json and img/video
    save_path = result_path + "json/"
    img_path = result_path + "img_video/"
    
    # if epoch^=None, append epoch in path
    if epoch != None:
        save_path = save_path + 'epoch_' + str(epoch) + '/'
        make_dir(save_path)
        img_path = img_path + 'epoch_' + str(epoch) + '/'
        make_dir(img_path)
    
    # if epoch=True, append best in path
    if best == True:
        save_path = save_path + 'best/'
        make_dir(save_path)
        img_path = img_path + 'best/'
        make_dir(img_path)
    
    # Predict
    predictions = model.predict([X_, decoder_input_array])
    dtw_dict = {}
    
    for i in range(len(X_)):
        # Slice predict array as frame count
        leng = output_skels[i]
        predict = predictions[i].tolist()[:leng]
        
        # if DTW Score calculated
        # Save filename with DTW Score
        try:
            score_dtw = calculate_dtw(y[i], predict)
            dtw_dict[str(output_file[i]) + '_' + str(output_gloss[i])] = score_dtw
            dtw_path = "{}".format("{0:.2f}".format(float(score_dtw)).replace(".", "_"))
            filename = str(output_file[i]) + '_' + str(output_gloss[i]) + '_' + dtw_path + '.json'   
            with open(save_path + filename, 'w', encoding='utf-8') as make_file: 
                json.dump(predict, make_file, indent="\t")
        
        # if DTW Score not calculated
        except:
            print("Cannot calcuate DTW Score")
            filename = str(output_file[i]) + '_' + str(output_gloss[i]) + '.json'       
            with open(save_path + filename, 'w', encoding='utf-8') as make_file: 
                json.dump(predict, make_file, indent="\t") 

        # Make img and video file
        create_img_video(save_path, img_path, filename)
    
    return