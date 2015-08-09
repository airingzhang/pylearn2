'''
Created on Jul 15, 2015

@author: ningzhang
'''
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.space import CompositeSpace
from _ast import Dict
from collections import OrderedDict
"""
This module tests multimodal_dbn 
"""

import os
import numpy as np
import theano
from pylearn2.testing import skip
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse
from pylearn2.utils import serial, data_writer
from pylearn2.datasets.flickr_image_toronto import Flickr_Image_Toronto
from pylearn2.datasets.flickr_text_toronto import Flickr_Text_Toronto
from pylearn2.datasets.composite_dataset import CompositeDataset

@no_debug_mode
def train_yaml(yaml_file):

    train = yaml_parse.load(yaml_file)
    train.main_loop()


def train_image_layer1(yaml_file_path, save_path):

    yaml = open("{0}/image_rbm1.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'batch_size': 100,
                    'monitoring_batches': 100,
                    'nhid': 2048,
                    'num_chains': 100,
                    'num_gibbs_steps': 1,
                    'max_epochs': 10000,
                    'save_path': save_path}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def train_image_layer2(yaml_file_path, save_path):

    yaml = open("{0}/image_rbm2.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'nvis': 2048,
                    'batch_size': 100,
                    'monitoring_batches': 100,
                    'nhid': 1024,
                    'num_chains': 100,
                    'num_gibbs_steps': 1,
                    'max_epochs': 10000,
                    'save_path': save_path}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)
    
def train_text_layer1(yaml_file_path, save_path):

    yaml = open("{0}/text_rbm1.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'batch_size': 100,
                    'monitoring_batches': 100,
                    'nhid': 2048,
                    'num_chains': 100,
                    'num_gibbs_steps': 1,
                    'max_epochs': 10000,
                    'save_path': save_path}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def train_text_layer2(yaml_file_path, save_path):

    yaml = open("{0}/text_rbm2.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'nvis': 2048,
                    'batch_size': 100,
                    'monitoring_batches': 100,
                    'nhid': 1024,
                    'num_chains': 100,
                    'num_gibbs_steps': 1,
                    'max_epochs': 10000,
                    'save_path': save_path}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)

def get_representations_for_joint_layer(yaml_file_path, save_path, batch_size):
    """
    The purpose of doing this is to test the compatibility of DBM with StackBlocks and TransformerDatasets
    Of course one can instead take use the "get_represenations.py" for data preparation for the next step.
    """
    
    hyper_params = {'save_path':save_path}
    yaml = open("{0}/stacked_image_unimodaliy.yaml".format(yaml_file_path), 'r').read()
    yaml = yaml % (hyper_params)
    image_stacked_blocks = yaml_parse.load(yaml)
    yaml = open("{0}/stacked_text_unimodaliy.yaml".format(yaml_file_path), 'r').read()
    yaml = yaml % (hyper_params)
    text_stacked_blocks = yaml_parse.load(yaml)
      
    image_raw = Flickr_Image_Toronto(which_cat = 'unlabelled',which_sub='nnz', using_statisfile = True)
    image_rep = TransformerDataset( raw = image_raw, transformer = image_stacked_blocks )
    m, n = image_raw.get_data().shape()
    dw = data_writer.DataWriter(['image_h2_rep'], save_path + 'image/', '10G', [n], m)
    image_iterator = image_rep.iterator(batch_size= batch_size)
    
    for data in image_iterator:
        dw.Submit(data)
    dw.Commit()
    text_raw = Flickr_Text_Toronto(which_cat='unlabelled')
    text_rep = TransformerDataset( raw = text_raw, transformer = text_stacked_blocks )
    m, n = text_raw.get_data().shape()
    dw = data_writer.DataWriter(['text_h2_rep'], save_path + 'text/', '10G', [n], m)
    text_iterator = text_rep.iterator(batch_size= batch_size)
    
    for data in text_iterator:
        dw.Submit(data)
    dw.Commit()
    
def inference_for_missing_data(yaml_file_path, save_path, batch_size, gibbs_steps, niter = 1):
    
    """
    The purpose of doing this:
    1. test if the composite visible and hidden layer work.
    2. test the inference of missing modality
    Todo: here we named the modal as DBN, however it is be a DBM.
    But we has set niter as 1, then there no difference between DBN and DBM for this test.
    One need to implement a DBN version upward_pass for DBN.
    Also the MF inference procedure, down_pass and some other part should be able to adapt to DBN style.
    
    """
    #modal
    hyper_params = {'save_path':save_path}
    yaml = open("{0}/multimodal_dbn.yaml".format(yaml_file_path), 'r').read()
    yaml = yaml % (hyper_params)
    multimdal_blocks = yaml_parse.load(yaml)
    
    """
    It is better to set the normalized_D in Replicated SoftMax visible layer
    because in this scenario, we can have a D set because text modality is missing.
    
    As in Nitish's paper, normalized_D is set to be 5. 
    Furthermore, it would be better to set normalized_D during the pretraining of text-pathway.
    
    Here we just bypass this step first in this test
    """
    
    # dataset
    image_raw = Flickr_Image_Toronto(which_cat = 'unlabelled',which_sub='z', using_statisfile = True)

    m, n = image_raw.get_data().shape()
    dw = data_writer.DataWriter(['missing data'], save_path + 'inferred_text/', '10G', [n], m)
    image_iterator = image_raw.iterator(batch_size= batch_size)
    
    text_input_space = multimdal_blocks.layers()[0].components[1].get_input_space()
    text_zeros = np.zeros((batch_size, text_input_space.get_total_dimension()))
    
    """Inference procedure, referring to the paper from Michigan:
    improved Multimodal Deep Learning with Variation of Information
    """
    
    image_tensor = multimdal_blocks.layers()[0].components[0].get_input_space().make_theano_batch()
    text_tensor = text_input_space.make_theano_batch()
    X = CompositeSpace(image_tensor,text_tensor)
    
    """
    Here to distinguish from DBN with DBM, do not use the __call__ upward_pass or function,
    instead, retrieve all layers of StackedBlock handly one by one.
    """
    # up_massage = multimdal_blocks(X, niter = niter)
    up_massage = [X]
    for i, layer in enumerate(multimdal_blocks.layers()):
        # here we use DBN, therefore no need to double the weights of intermediate layer(s)
        outputs = layer.upward_pass(v = up_massage[-1],niter = niter, double_bottom = False)
        up_massage.append(outputs)

    
    clamped_state = up_massage[-2]
    layer_to_state = OrderedDict()
        
    layer_to_state[multimdal_blocks.layers()[-1].visible_layer]  = clamped_state
    layer_to_state[multimdal_blocks.layers()[-1].hidden_layers[0]]  = up_massage[-1]
    
    #gibbs sampling 
    for i in xrange(gibbs_steps):
        # one step
        layer_to_state = multimdal_blocks.layers()[-1].sampling_procedure(layer_to_state)
        #clamp to the image representation
        layer_to_state[multimdal_blocks.layers()[-1].visible_layer[0]]  = clamped_state[0]
        
    #downward passing 
    input_state = list(layer_to_state[multimdal_blocks.layers[-1].visible_layer])
    for i in xrange(0,len(multimdal_blocks.layers)-1):
        input_state = multimdal_blocks.layers[len(multimdal_blocks.layers)-2-i].downward_pass(input_state = input_state, niter = niter,double_top = False)
    
    fn = theano.function([X],input_state[1])    
       
    for data in image_iterator:
        total_input  = [data,text_zeros]
        res = fn(total_input)
        dw.Submit(res)
    dw.Commit()
        
def train_joint_hidden_layer(yaml_file_path, save_path):
    
    yaml = open("{0}/joint_hidden_layer.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'X_dir_1': save_path + 'image/',
                    'X_dir_2': save_path + 'text/',
                    'batch_size': 100,
                    'monitoring_batches': 100,
                    'nhid': 1024,
                    'num_chains': 100,
                    'num_gibbs_steps': 1,
                    'max_epochs': 10000,
                    'save_path': save_path}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def train_(yaml_file_path, save_path):

    yaml = open("{0}/multimodal_dbn.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'train_stop': 50,
                    'valid_stop': 50050,
                    'batch_size': 50,
                    'max_epochs': 1,
                    'save_path': save_path}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def test_multimodal_dbn():

    skip.skip_if_no_data()

    yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  '..'))
    save_path = serial.preprocess("${PYLEARN2_DATA_PATH}/experiment/multimodal_dbn/")

    train_image_layer1(yaml_file_path, save_path)
    train_image_layer2(yaml_file_path, save_path)
    train_text_layer1(yaml_file_path, save_path)
    train_text_layer2(yaml_file_path, save_path)
    
    get_representations_for_joint_layer(yaml_file_path, save_path, 100)
    train_joint_hidden_layer(yaml_file_path, save_path)
    
 #   train_mlp(yaml_file_path, save_path)

    try:
        os.remove(save_path +"image_rbm1.pkl".format(save_path))
        os.remove(save_path +"image_rbm2.pkl".format(save_path))
    except OSError:
        pass

if __name__ == '__main__':
    test_multimodal_dbn()
