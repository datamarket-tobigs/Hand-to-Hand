import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow.keras.backend as K
import numpy as np
import pickle
import json

from data.data import load_data
from modeling.model import build_model
from modeling.helpers import  load_config, make_dir
from modeling.prediction import make_predict
from modeling.key2video import create_img_video

"""
Set callbacks and execute training process.
"""

class MyCallback(tf.keras.callbacks.Callback):
    """
    Set the custom callback class
    to print learning rate every 5 epochs and save model and prediction every 10 epochs.
    (Given periods are just for sample data(train, dev, test, each set has 5).
    Freely change values as you want.)

    :method __init__: call required parameter
    :method on_epoch_end: execute at the end of each epoch
    """
    
    def __init__(self, name, cfg, X, y, decoder_input_array,
                 output_file, output_gloss, output_skels):
        """
        Call required parameters.

        :param inheritance from default keras callbacks(ex. epoch)
        :param self.cfg: configuration dictionary
        :param self.X: X_dev from data.py
        :param self.y: y_dev from data.py
        :param self.decoder_input_array: decoder_input_array_dev from data.py
        :param self.output_file: output_file_dev
        :param self.output_gloss: output_gloss_dev
        :param self.output_skels: output_skels_dev
        """
        super().__init__() #inheritance from tf.keras.callbacks.Callback
        # additional params
        self.cfg = cfg
        self.X = X
        self.y = y
        self.decoder_input_array = decoder_input_array
        self.output_file = output_file
        self.output_gloss = output_gloss
        self.output_skels = output_skels
        
    def on_epoch_end(self, epoch, logs=None):
        """
        executed in the end of every epochs.
        """
        # Print learning rate every 5 epochs
        if epoch > 0 and epoch % 5 == 0:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            print('learning rate : ', lr)

        # Save model and prediction every 10 epochs
        if epoch > 0 and epoch % 10 == 0:
            model_path = self.cfg["model_path"]
            self.model.save(model_path+"model.h5")
            result_path = self.cfg["result_path"]
            make_predict(self.cfg, self.model, self.X, self.y, self.decoder_input_array,
                         self.output_file, self.output_gloss, self.output_skels,
                         result_path, epoch, best=False)
  
            
def Train(cfg: dict) -> None:
    """
    Execute train process with the base configs.
    
    :param cfg: configuration dictionary (Base.yaml)
    """
    # Load train, dev data 
    X_train, y_train, decoder_input_array_train, mel_spectro_data_array_train, max_X, vocab_size_source = load_data(cfg=cfg, mode="train")
    X_dev, y_dev, decoder_input_array_dev, mel_spectro_data_array_dev = load_data(cfg=cfg, mode="dev")
    print("---------------------------------------------------")
    print("Complete: Load train, dev data")
    print("---------------------------------------------------")
    
    # Make result directories
    model_path = cfg["model_path"]
    make_dir(model_path) #"./Models/"
    result_path = cfg["result_path"]
    make_dir(result_path) # "./Models/result/"
    print("---------------------------------------------------")
    print("Complete: Make result directories")
    print("---------------------------------------------------")
     
    # Save real json, img, video before training
    json_path = result_path + "json/"
    make_dir(json_path) # "./Models/result/json/"
    img_path = result_path + "img_video/"
    make_dir(img_path) # "./Models/result/"img_video/"

    data_path = cfg["data_path"] 
    with open(data_path + 'out_files_dev' +'.pickle', 'rb') as f:
        output_file = pickle.load(f)
    with open(data_path + 'out_gloss_dev' +'.pickle', 'rb') as f:
        output_gloss = pickle.load(f)
    with open(data_path + 'out_skels_dev' +'.pickle', 'rb') as f:
        output_skels = pickle.load(f)
    
    real_json_path = json_path + 'real/'
    make_dir(real_json_path)
    real_img_path = img_path + 'real/'
    make_dir(real_img_path)
    
    for i in range(len(X_dev)):
        leng = output_skels[i]
        real = y_dev[i].tolist()[:leng]
        filename = str(output_file[i]) + '_' + str(output_gloss[i]) + '_real' + '.json'
        
        with open(real_json_path + filename, 'w', encoding='utf-8') as make_file: 
            json.dump(real, make_file, indent="\t")
        
        #make img & video
        create_img_video(real_json_path, real_img_path, filename)
    print("---------------------------------------------------")
    print("Complete: Save real json, img and video files")
    print("---------------------------------------------------")
    
    # Build the tacotron model
    model = build_model(cfg=cfg, max_X=max_X, vocab_size_source=vocab_size_source)
    print("---------------------------------------------------")
    print("Complete: Build model")
    print("---------------------------------------------------")
    
    # Set Optimizer(Adam) and Loss(MSE)
    opt = Adam()
    model.compile(optimizer=opt,
                loss=['mean_squared_error', 'mean_squared_error']) # original was 'mean_absolute_error'

    # Set Callback options
    ### callback1: customized callback (save model and make prediction every 1000 epochs)
    first_callback = MyCallback('save_jsonfile', cfg, X_dev, y_dev, decoder_input_array_dev,
                                output_file, output_gloss, output_skels)

    ### callback2: best model save (update best model.h5 every 10 epochs)
    best_path = model_path + "best_model.h5"

    best_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        period=10)

    ### callback3: learning rate scheduler (reduce LR by 20% when there is no enhancement of val_loss every 100 epochs)
    patience = cfg["training"].get("patience", 10)
    decrease_factor = cfg["training"].get("decrease_factor", 0.2)
    min_LR = cfg["training"].get("min_LR", 0.00001)

    reduceLR = ReduceLROnPlateau(
        monitor='val_loss',
        factor=decrease_factor,
        patience=patience,
        min_lr=min_LR)

    ### (optional callback)
    # 1. early stopping
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience = 20)

    
    print("---------------------------------------------------")
    print("Start training!")
    print("---------------------------------------------------")

    # Fit Model
    batch_size = cfg["training"].get("batch_size", 2)
    epochs = cfg["training"].get("epoch", 100)

    train_history = model.fit([X_train, decoder_input_array_train],
                            mel_spectro_data_array_train,
                            epochs=epochs, batch_size=batch_size, shuffle=False,
                            verbose=1,
                            validation_data=([X_dev, decoder_input_array_dev], mel_spectro_data_array_dev),
                            callbacks = [first_callback, best_callback, reduceLR]) #total 3 callbacks
    
    print("---------------------------------------------------")
    print("Finish Training! Save the last model and prediction.")
    print("---------------------------------------------------")

    # Save the last Model(100 epoch) and prediction
    model.save(model_path + 'model.h5')
    make_predict(cfg, model, X_dev, y_dev, decoder_input_array_dev,
                 output_file, output_gloss, output_skels, result_path, epochs, best=False)
    
    print("---------------------------------------------------")
    print("Congrats! All works well~!")
    print("---------------------------------------------------")


