from tensorflow.keras.models import Model
from .tacotron import *


def build_model(cfg: dict, max_X, vocab_size_source):
    """
    Build Tacotron model with base architectures from tacotron.py.    
    
    :param cfg: configuration dictionary
    :param max_X: max length of X_train and X_dev 
    :param vocab_size_source: size of vocabulary dictionary
    :return model: model with given input and output form
    """
    # Bring required materials from cfg dicionary
    full_cfg = cfg
    cfg = cfg["model"]

    latent_dim = cfg["latent_dim"]
    K1 = cfg["K1"]
    N_MEL = cfg["N_MEL"]
    vocab_size_target = cfg["vocab_size_target"]
    MAX_MEL_TIME_LENGTH = cfg["MAX_MEL_TIME_LENGTH"]
    r = cfg["r"]
    
    # Bring <encoder> from tacotron.py
    encoder = Encoder(max_X, latent_dim, vocab_size_source, K1)
    input_encoder, cbhg_encoding = encoder(0, 0)

    # Bring <decoder prenet> from tacotron.py
    decoder_prenet = Decoder_prenet(N_MEL, latent_dim, vocab_size_target)
    input_decoder, attention_rnn_output = decoder_prenet(0, 0)
    
    # Bring <attention> from tacotron.py
    attention = Attention(max_X, cbhg_encoding, attention_rnn_output)
    attention_context, attention_rnn_output_reshaped = attention(0, 0)

    # Bring <decoder> from tacotron.py
    decoder = Decoder(attention_context,
                    attention_rnn_output_reshaped, MAX_MEL_TIME_LENGTH, N_MEL, r) # r 추가
    mel_hat_ = decoder(0) 

    # Build model
    model = Model([input_encoder, input_decoder], outputs=mel_hat_)
    
    return model