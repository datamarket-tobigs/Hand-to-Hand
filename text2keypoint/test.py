import tensorflow as tf
import os
import pickle

from data.data import load_data
from modeling.prediction import make_predict
from modeling.helpers import make_dir

def Test(cfg: dict) -> None:
    """
    Execute test process with the base configs.
    
    :param cfg: configuration dictionary (Base.yaml)
    """

    # Load the test data
    X_test, y_test, decoder_input_array_test, mel_spectro_data_array_test = load_data(cfg=cfg, mode="test")
    print("---------------------------------------------------")
    print("Complete: Load test data")
    print("---------------------------------------------------")

    # Load preprocessing data(output_file, output_gloss, output_skels)
    path= cfg["data_path"]
    with open(path + 'out_files_test' +'.pickle', 'rb') as f:
        output_file = pickle.load(f)
    with open(path + 'out_gloss_test' +'.pickle', 'rb') as f:
        output_gloss = pickle.load(f)
    with open(path + 'out_skels_test' +'.pickle', 'rb') as f:
        output_skels = pickle.load(f)
    
    # Make test result directory
    result_path = cfg["test_result_path"]
    make_dir(result_path) # "./test_result/"
        
    save_path = result_path + "json/"
    make_dir(save_path) # "./test_result/json/"
    img_path = result_path + "img_video/"
    make_dir(img_path) # "./test_result/img_video/"
    print("---------------------------------------------------")
    print("Complete: Make test_result directories")
    print("---------------------------------------------------")
        
    # Load Model(best or recent)
    test_mode = cfg["test_mode"]
    
    if test_mode == "best":
        best_model_path = cfg["model_path"] + "best_model.h5"
        model = tf.keras.models.load_model(best_model_path) # best model load
        print("---------------------------------------------------")
        print("Complete: Load best model")
        print("---------------------------------------------------")
        # Make prediction files(json and img, video)
        make_predict(cfg, model, X_test, y_test, decoder_input_array_test,
                     output_file, output_gloss, output_skels,
                     result_path, epoch=None, best=True)
    
    elif test_mode == "recent":
        recent_model_path = cfg["model_path"] + "model.h5"
        model = tf.keras.models.load_model(recent_model_path) # most recent model load
        print("---------------------------------------------------")
        print("Complete: Load recent model")
        print("---------------------------------------------------")
        # Make prediction files(json and img, video)
        make_predict(cfg, model, X_test, y_test, decoder_input_array_test,
                     output_file, output_gloss, output_skels,
                     result_path, epoch=None, best=False)
    
    print("---------------------------------------------------")
    print("Complete: Save prediction json, img and video files")
    print("---------------------------------------------------")