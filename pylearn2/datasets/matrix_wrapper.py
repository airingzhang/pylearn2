'''
Created on Aug 3, 2015

@author: ningzhang
'''

import glob
import numpy
import pylearn2.utils.serial as serial
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets import control, cache
from pylearn2.datasets.exc import NotInstalledError

class DenseDesignMartixWrapper(DenseDesignMatrix):
    '''
    This is a simple wrapper for DenseDesignMatrix which can read data from files.
    Dir: directory of files 
    
    '''
    _default_seed = (17, 2, 946)
    
    def __init__(self, X_dir, topo_view=None, y_dir=None,
                 view_converter=None, axes=('b', 0, 1, 'c'),
                 rng=_default_seed, preprocessor=None, fit_preprocessor=False,
                 X_labels=None, y_labels=None):
        '''
        Constructor
        '''
        if X_dir is None:
            raise ValueError("X_dir should not be None when using DenseDesignMartixWrapper")
        
        im_path = serial.preprocess(X_dir)
        files = glob.glob(im_path + '*.npy')  
        datasetCache = cache.datasetCache
        # Locally cache the files before reading them
        initialized = False
        m_total = 0
        for f in files:
            im_path = datasetCache.cache_file(f)
            try:
                temp = serial.load(im_path)
            except IOError:
                raise NotInstalledError("Flickr_imgae_Toronto image files cannot be "
                                                    "found in " + im_path + ". Please check the directory")
            m, d = temp.shape
            assert d == 3857
            m_total += m 
            if not initialized :
                X = temp
                initialized = True
            else:
                numpy.concatenate((X, temp), axis=0)
        if y_dir is not None:
            im_path = serial.preprocess(y_dir)
            files = glob.glob(im_path + '*.npy')  
            datasetCache = cache.datasetCache
            # Locally cache the files before reading them
            initialized = False
            m_total = 0
            for f in files:
                im_path = datasetCache.cache_file(f)
                try:
                    temp = serial.load(im_path)
                except IOError:
                    raise NotInstalledError("Flickr_imgae_Toronto image files cannot be "
                                                        "found in " + im_path + ". Please check the directory")
                m, d = temp.shape
                assert d == 3857
                m_total += m 
                if not initialized :
                    y = temp
                    initialized = True
                else:
                    numpy.concatenate((y, temp), axis=0)        
        else:
            y = None
            
        super(DenseDesignMartixWrapper, self).__init__(
                X=X, y = y, topo_view=topo_view,
                view_converter=view_converter, axes=('b', 0, 1, 'c'),
                rng=rng, preprocessor=preprocessor, fit_preprocessor=fit_preprocessor,
                X_labels=X_labels, y_labels=y_labels
            )
        
