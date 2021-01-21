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
