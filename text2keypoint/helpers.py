import yaml
import os
import tensorflow as tf

"""
Several functions to help overall modeling process 
"""

########################################################################

def load_config(path="configs/Base.yaml") -> dict:
# reference: https://github.com/BenSaunders27/ProgressiveTransformersSLP
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return cfg: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

########################################################################

def make_dir(path):
    """
    Make a new directory in given path.

    :param path: path to new directory
    """
    if not os.path.exists(path):
        os.mkdir(path)
    return

########################################################################

def Max_length(data):
    """
    Get the max lengh of given data.

    :param data: list of sentences(str)
    :return max_length_: max length of words in sentences
    """
    max_length_ = max([len(x.split(' ')) for x in data])
    return max_length_

########################################################################

def seq2text(input_seq, index2word):
    """
    Change sequence to Word ([1,2,4] -> ['I','love','dog])
    
    :param input_seq: word to sequence
    :param index2word: Dict(key:index, value:word)
    :return newString: Word from input_seq
    """
    newString = ''
    for i in input_seq:
      if(i != 0):
        newString = newString+index2word[i]+' '
    return newString

########################################################################

