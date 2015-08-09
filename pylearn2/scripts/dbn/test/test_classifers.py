'''
Created on Aug 9, 2015

@author: ningzhang
'''

import os
import numpy as np
import glob as glob

import theano
from pylearn2.testing import skip
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse
from pylearn2.testing import skip
import pylearn2.utils.performance_metric
import pylearn2.utils.serial as serial 
from pylearn2.models.tests.test_autoencoder import yaml_dir_path

def represenations_dealler():
    """
    This method is to split and combine representations for different
    classifiers trained with different representations from layers of 
    mutilmodal DBN.
    
    """
    
    # first deal with joint layer representations 
    
    dir = "{}"
    dir = serial.preprocess(dir)
    files = glob.glob(dir+"*.npy")
    
@no_debug_mode
def train_yaml(yaml_file):

    train = yaml_parse.load(yaml_file)
    train.main_loop()


def test_joint_layer_classifer(yaml_file_path, save_path):
    """
    Train a logistic regression model beyond the joint hidden layer
    
    """
    
    yaml = open("{0}/classifer/joint_layer_classifier.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'X_dir': "",
                    'y_dir': "",
                    'nvis': 2048,
                    'save_path': save_path}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)    

    
def main():
    
    yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  '..'))
    save_path = serial.preprocess("${PYLEARN2_DATA_PATH}/experiment/multimodal_dbn/")

if __name__ == '__main__':
    main()