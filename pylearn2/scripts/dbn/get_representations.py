'''
Created on Aug 1, 2015

@author: ningzhang
'''
import sys
import numpy as np 

import theano
import theano.tensor as T

from pylearn2.utils import data_writer
from pylearn2.utils import serial
import pylearn2.datasets.flickr_image_toronto as flickr_image_toronto
import pylearn2.datasets.flickr_text_toronto as flickr_text_toronto



def getImageRepresenations(model_path, output_dir, from_all_layers = False):
    
    try:
        print "Loading model"
        model = serial.load(model_path)
    except Exception as e:
        print "error loading {}:".format(model_path)
        print e
        return False
    
    """
    Actually this can be optimized to avoid loading all files at one time which may consume 
    quite a lot memory 
    Here I just save some time by directly take use of dataset module
    """
    raw_data = flickr_image_toronto.Flickr_Image_Toronto(which_cat = "unlabelled", using_statisfile = True).get_data()
 
    X = model.get_input_space().make_theano_batch()
    batch_size = model.batch_size
    batch_num = raw_data.shape(0) / batch_size
    extend = raw_data.shape(0) - batch_num * batch_size
    if extend:
        batch_num = batch_num +1
        last_bacth = raw_data[batch_num * batch_size:-1,:].append(np.zeros((extend, raw_data.shape(1))))
    
    numlist = []
    dimlist = []       
    if from_all_layers:
        H_hat = model.inference_procedure.mf(V = X, niter=1)
        rval = []
        for i, layer in enumerate(model.hidden_layers):
            numlist.add(output_dir + "flickr_image_h%d" %i)
            dimlist.add(layer.ndim)
            rval.append(model.hidden_layers[i].upward_state(H_hat[i]))
        dw = data_writer.DataWriter(numlist, output_dir, "10G", dimlist, raw_data.shape(0))    
        fn = theano.function([X], rval, name='perform_full')
        
    else:
        numlist.add(output_dir +"flickr_image_h%d" % len(model.hidden_layers))
        dimlist.add(model.hidden_layers[-1].ndim)
        H_hat = model.inference_procedure.mf(X,niter=1)
        rval = model.hidden_layers[-1].upward_state(H_hat[-1])
        fn = theano.function([X], rval, name='perform')
        
    for i in xrange(batch_num):
        rep = fn(raw_data[i * batch_size: (i+1) * batch_size -1, :]) 
        dw.Submit(rep) 
    if extend:
        rep = fn(last_bacth)    
        dw.Submit(rep) 
    dw.commit()
    
def getTextRepresenations(model_path, output_dir, from_all_layers = False):
    """
    Pay attention the sparse property of raw data  
    """
    try:
        print "Loading model"
        model = serial.load(model_path)
    except Exception as e:
        print "error loading {}:".format(model_path)
        print e
        return False
    
    raw_data = flickr_text_toronto.Flickr_Text_Toronto(which_cat = "unlabelled").get_data()
 
    X = model.get_input_space().make_theano_batch()
    batch_size = model.batch_size
    batch_num = raw_data.shape(0) / batch_size
    extend = raw_data.shape(0) - batch_num * batch_size
    if extend:
        batch_num = batch_num +1
        last_bacth = raw_data[batch_num * batch_size:-1,:].full().append(np.zeros((extend, raw_data.shape(1))))
    
    numlist = []
    dimlist = []       
    if from_all_layers:
        H_hat = model.inference_procedure.mf(X,niter=1)
        rval = []
        for i, layer in enumerate(model.hidden_layers):
            numlist.add(output_dir + "flickr_text_h%d" %i)
            dimlist.add(layer.ndim)
            rval.append(model.hidden_layers[i].upward_state(H_hat[i]))
        
        dw = data_writer.DataWriter(numlist, output_dir, "10G", dimlist, raw_data.shape(0))    
        fn = theano.function([X], rval, name='perform_full')
        
    else:
        numlist.add(output_dir +"flickr_text_h%d" % len(model.hidden_layers) )
        dimlist.add(model.hidden_layers[-1].ndim)
        H_hat = model.inference_procedure.mf(X,niter=1)
        rval = model.hidden_layers[-1].upward_state(H_hat[-1])
        fn = theano.function([X], H_hat, name='perform')
        
    for i in xrange(batch_num):
        rep = fn(raw_data[i * batch_size: (i+1) * batch_size -1, :].full()) 
        dw.Submit(rep) 
    if extend:
        rep = fn(last_bacth)    
        dw.Submit(rep) 
        
    dw.commit()

def main():
    
    base = "${PYLEARN2_DATA_PATH}/flickr/image/"
    output_dir = serial.preprocess(base)
    model_path = "${PYLEARN2_DATA_PATH}/experiment/multimodal_dbn/"
    model_path  = serial.preprocess(model_path)
    getImageRepresenations(model_path, output_dir, from_all_layers = True)
    getTextRepresenations(model_path, output_dir, from_all_layers = True)
    
    
    
    
    

if __name__ == '__main__':
    main()
