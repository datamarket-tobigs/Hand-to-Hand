import tensorflow as tf
import os
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow.keras.initializers as k_init
from tensorflow.keras.layers import Input, Embedding, concatenate, RepeatVector, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, Dense, Activation, MaxPooling1D, Add,
                          Concatenate, Bidirectional, GRU, Dropout,
                          BatchNormalization, Lambda, Dot, Multiply)
from tensorflow.keras.layers import Input, Embedding, concatenate, RepeatVector, Dense, Reshape

"""
Base architecture of Tacotron (Encoder, Decoder prenet, Attention, Decoder)

code reference: https://www.wolf.university/hands-onnaturallanguageprocessingwithpython/ebook/hands-onnaturallanguageprocessingwithpython.pdf
"""

# Encoder

class Encoder(tf.keras.Model):

  def __init__(self, max_X, latent_dim, vocab_size_source, k1):
    super(Encoder, self).__init__()

    self.k1 = k1
    self.max_X = max_X
    self.latent_dim = latent_dim
    self.vocab_size_source = vocab_size_source

    self.enc_emb = tf.keras.layers.Embedding(
        vocab_size_source, latent_dim, trainable=True, mask_zero=True)

  def get_pre_net(self, input_data):

    prenet = Dense(256)(input_data)
    prenet = Activation('relu')(prenet)
    prenet = Dropout(0.5)(prenet)
    prenet = Dense(128)(prenet)
    prenet = Activation('relu')(prenet)
    prenet = Dropout(0.5)(prenet)

    return prenet

  def get_conv1dbank(self, K_, input_data):
    conv = Conv1D(filters=128, kernel_size=1,
                  strides=1, padding='same')(input_data)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    for k_ in range(2, K_ + 1):
        conv = Conv1D(filters=128, kernel_size=k_,
                      strides=1, padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

    return conv

  def get_highway_output(self, highway_input, nb_layers, activation="tanh", bias=-3):

    dim = K.int_shape(highway_input)[-1]  # dimension must be the same
    initial_bias = k_init.Constant(bias)
    for n in range(nb_layers):
        H = Dense(units=dim, bias_initializer=initial_bias)(highway_input)
        H = Activation("sigmoid")(H)
        carry_gate = Lambda(lambda x: 1.0 - x,
                            output_shape=(dim,))(H)
        transform_gate = Dense(units=dim)(highway_input)
        transform_gate = Activation(activation)(transform_gate)
        transformed = Multiply()([H, transform_gate])
        carried = Multiply()([carry_gate, highway_input])
        highway_output = Add()([transformed, carried])
    return highway_output

  def get_CBHG_encoder(self, input_data, K_CBHG):

    conv1dbank = self.get_conv1dbank(K_CBHG, input_data)
    conv1dbank = MaxPooling1D(pool_size=2, strides=1,
                              padding='same')(conv1dbank)
    conv1dbank = Conv1D(filters=128, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)
    conv1dbank = Activation('relu')(conv1dbank)
    conv1dbank = Conv1D(filters=128, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)

    residual = Add()([input_data, conv1dbank])

    highway_net = self.get_highway_output(residual, 4, activation='relu')

    CBHG_encoder = Bidirectional(GRU(128, return_sequences=True))(highway_net)

    return CBHG_encoder

  def call(self, i, j):
    input_encoder = Input(shape=(self.max_X,))
    input_encoder_ = self.enc_emb(input_encoder)
    prenet_encoding = self.get_pre_net(input_encoder_)
    cbhg_encoding = self.get_CBHG_encoder(prenet_encoding, self.k1)

    return input_encoder, cbhg_encoding

# Decoder prenet

class Decoder_prenet(tf.keras.Model):

  def __init__(self, n_mels, latent_dim, vocab_size_target):
    super(Decoder_prenet, self).__init__()

    self.n_mels = n_mels
    self.latent_dim = latent_dim
    self.vocab_size_target = vocab_size_target

    self.dec_emb = tf.keras.layers.Embedding(
        vocab_size_target, latent_dim, trainable=True, mask_zero=True)

  def get_pre_net(self, input_data):

    prenet = Dense(256)(input_data)
    prenet = Activation('relu')(prenet)
    prenet = Dropout(0.5)(prenet)
    prenet = Dense(128)(prenet)
    prenet = Activation('relu')(prenet)
    prenet = Dropout(0.5)(prenet)

    return prenet

  def get_attention_RNN(self):
    return GRU(256)

  def call(self, i, j):
    input_decoder = Input(shape=(None, self.n_mels))
    input_decoder_ = self.dec_emb(input_decoder)

    prenet_decoding = self.get_pre_net(input_decoder)
    attention_rnn_output = self.get_attention_RNN()(prenet_decoding)

    return input_decoder, attention_rnn_output

# Attention

class Attention(tf.keras.Model):

  def __init__(self, max_X, cbhg_encoding, attention_rnn_output):
      super(Attention, self).__init__()
      self.max_X = max_X
      self.cbhg_encoding = cbhg_encoding
      self.attention_rnn_output = attention_rnn_output

  def get_attention_context(self, encoder_output, attention_rnn_output):
      attention_input = Concatenate(axis=-1)([encoder_output,
                                              attention_rnn_output])
      e = Dense(10, activation="tanh")(attention_input)
      energies = Dense(1, activation="relu")(e)
      attention_weights = Activation('softmax')(energies)
      context = Dot(axes=1)([attention_weights,
                             encoder_output])

      return context

  def call(self, i, j):
      attention_rnn_output_repeated = RepeatVector(
          self.max_X)(self.attention_rnn_output)
      #attention_rnn_output from decoder

      attention_context = self.get_attention_context(
          self.cbhg_encoding, attention_rnn_output_repeated)
      #cbhg_encoding from encoder

      context_shape1 = int(attention_context.shape[1])
      context_shape2 = int(attention_context.shape[2])
      attention_rnn_output_reshaped = Reshape(
          (context_shape1, context_shape2))(self.attention_rnn_output)

      return attention_context, attention_rnn_output_reshaped

# Decoder

class Decoder(tf.keras.Model):

  def __init__(self, attention_context, attention_rnn_output_reshaped, mel_time_length, n_mels, r):
    super(Decoder, self).__init__()
    
    self.r = r
    self.n_mels = n_mels
    self.mel_time_length = mel_time_length
    self.attention_context = attention_context
    self.attention_rnn_output_reshaped = attention_rnn_output_reshaped


  def get_decoder_RNN_output(self, input_data):

    rnn1 = GRU(256, return_sequences=True)(input_data)

    inp2 = Add()([input_data, rnn1])
    rnn2 = GRU(256)(inp2)

    decoder_rnn = Add()([inp2, rnn2])

    return decoder_rnn

  def get_conv1dbank(self, K_, input_data):
    conv = Conv1D(filters=128, kernel_size=1,
                  strides=1, padding='same')(input_data)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    for k_ in range(2, K_ + 1):
        conv = Conv1D(filters=128, kernel_size=k_,
                      strides=1, padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

    return conv
    
  def get_highway_output(self, highway_input, nb_layers, activation="tanh", bias=-3):
    
    dim = K.int_shape(highway_input)[-1]  # dimension must be the same
    initial_bias = k_init.Constant(bias)
    for n in range(nb_layers):
        H = Dense(units=dim, bias_initializer=initial_bias)(highway_input)
        H = Activation("sigmoid")(H)
        carry_gate = Lambda(lambda x: 1.0 - x,
                            output_shape=(dim,))(H)
        transform_gate = Dense(units=dim)(highway_input)
        transform_gate = Activation(activation)(transform_gate)
        transformed = Multiply()([H, transform_gate])
        carried = Multiply()([carry_gate, highway_input])
        highway_output = Add()([transformed, carried])

    return highway_output

  def get_CBHG_post_process(self, input_data, K_CBHG):

    conv1dbank = self.get_conv1dbank(K_CBHG, input_data)
    conv1dbank = MaxPooling1D(pool_size=2, strides=1,
                              padding='same')(conv1dbank)
    conv1dbank = Conv1D(filters=256, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)
    conv1dbank = Activation('relu')(conv1dbank)
    conv1dbank = Conv1D(filters=80, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)

    residual = Add()([input_data, conv1dbank])

    highway_net = self.get_highway_output(residual, 4, activation='relu')

    CBHG_encoder = Bidirectional(GRU(128))(highway_net)

    return CBHG_encoder

  def call(self, i):

    input_of_decoder_rnn = concatenate([self.attention_context, self.attention_rnn_output_reshaped])
    input_of_decoder_rnn_projected = Dense(256)(input_of_decoder_rnn)
    output_of_decoder_rnn = self.get_decoder_RNN_output(input_of_decoder_rnn_projected)
    
    mel_hat = Dense(self.mel_time_length * self.n_mels)(output_of_decoder_rnn)
    mel_hat_ = Reshape((self.mel_time_length, self.n_mels))(mel_hat)

    return mel_hat_