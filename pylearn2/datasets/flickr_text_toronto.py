'''
Created on Jul 9, 2015

@author: ningzhang
'''

"""
Preprocessed Flicr dataset from Toronto 
This dataset contains two main parts: image and text.
Data missing exist for this dataset. 
more details can be found from Nitish's paper: Multimodal Learning with Deep Boltzmann Machines  
URL: http://www.cs.toronto.edu/~nitish/multimodal/

"""
import glob
import numpy
from scipy import sparse as sp
from theano.compat.six.moves import xrange
from pylearn2.datasets.sparse_dataset import (
    SparseDataset
)
from pylearn2.datasets import control, cache
from pylearn2.datasets.exc import NotInstalledError
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng


class Flickr_Text_Toronto(SparseDataset):
    """
    Flickr image part, to make it easier to use for multimodal learning,
    we divide Flick dataset into 3 parts: image, text, and join.
    One can also refer to Flickr_Image_Toronto  and Flickr_Join_Toronto

    Parameters
    ----------
    which_cat : str
        Which category of the image dataset. Must be in ['labelled', 'unlabelled']
        If None, which_sub should not be none and it means both two categories.
    including_label: bool, optional
        Whether to including labels    
    shuffle : bool, optional
        Whether to shuffle the dataset. Defaults to `False`.
    start : int, optional
        Defaults to `None`. If set, excludes examples whose index is inferior
        to `start`.
    stop : int, optional
        Defaults to `None`. If set, excludes examples whose index is superior
        to `stop`.
    preprocessor : pylearn2.datasets.preprocessing.Preprocessor, optional
        Preprocessing to apply to the data. Defaults to `None`.
            
    """

    def __init__(self, which_cat, including_label = False, shuffle=False,
                 start=None, stop=None, preprocessor=None):
        self.args = locals()
        cat_options = ['labelled','unlabelled']
        if which_cat is None or which_cat not in cat_options:
            raise ValueError('Unrecognized which_cat value "%s".' %
                             (which_cat,) + '". Valid values are ' +
                             '["labelled", "unlabelled"].')
 
        path = "${PYLEARN2_DATA_PATH}/flickr/text/text_nnz_2000_"+ which_cat +".npz"
        im_path = cache.datasetCache.cache_file(path)
        try:
            npzfile = numpy.load(im_path)
        except IOError:
            raise NotInstalledError("Flickr_imgae_Toronto image files cannot be "
                                                    "found in " + im_path + ". Please check the directory")
        m, d = npzfile['shape']
        assert m == (which_cat == 'labelled' and 20449 or 851050)
        assert d == 2000

        X = sp.csr_matrix((npzfile['data'], npzfile['indices'],
                                  npzfile['indptr']),
                                  shape=tuple(list(npzfile['shape'])))    
        
        super(Flickr_Text_Toronto, self).__init__(
                from_scipy_sparse_dataset = X
        )  
        assert not numpy.any(numpy.isnan(self.X))  
        #=======================================================================
        # dealing with label if label is indicated to included by
        # which_set and which_sub
        #=======================================================================
        if including_label and which_cat != 'unlabelled':
        
            try:
                im_path = serial.preprocess("${PYLEARN2_DATA_PATH}/flickr/text/indices_labelled.npz")
                indices_labelled = numpy.load(im_path)
            except IOError:
                raise NotInstalledError("Flickr_imgae_Toronto indices_labelled file cannot be "
                                                "found in " + im_path + ". Please check the directory")
            try:
                im_path = serial.preprocess("${PYLEARN2_DATA_PATH}/flickr/labels.npy")
                labels =  numpy.load(im_path)
            except IOError:
                raise NotInstalledError("Flickr_imgae_Toronto label file cannot be "
                                                "found in " + im_path + ". Please check the directory")
            indices =indices_labelled['nnz_indices']          
            y = labels[indices[:]]     

            if shuffle:
                self.shuffle_rng = make_np_rng(None, [1, 2, 3],
                                           which_method="shuffle")
            
                X_lil = sp.lil_matrix(X)    
                for i in xrange(X.shape[0]):
                    j = self.shuffle_rng.randint(m)
                    # Copy ensures that memory is not aliased. together with label 
                    tmp = X_lil[i, :].copy()
                    X_lil[i, :] = X_lil[j, :]
                    X_lil[j, :] = tmp
                    tmp = y[i, :].copy()
                    y[i, :] = y[j, :]
                    y[j, :] = tmp
                
                X = sp.csr_matrix(X_lil) 
        else:
            if shuffle:
                self.shuffle_rng = make_np_rng(None, [1, 2, 3],
                                           which_method="shuffle")
                
                X_lil = sp.lil_matrix(X)     
                for i in xrange(X.shape[0]):
                    j = self.shuffle_rng.randint(m)
                    # Copy ensures that memory is not aliased.
                    tmp = X_lil[i, :].copy()
                    X_lil[i, :] = X_lil[j, :]
                    X_lil[j, :] = tmp
                X = sp.csr_matrix(X_lil) 
            
        #assert not numpy.any(numpy.isnan(self.X))    
        if start is not None and stop is not None:
            assert start >= 0
            if stop > self.X.shape[0]:
                raise ValueError('stop=' + str(stop) + '>' +
                                 'm=' + str(self.X.shape[0]))
            assert stop > start
            self.X = self.X[start:stop, :]
            if self.X.shape[0] != stop - start:
                raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                                 % (self.X.shape[0], start, stop))           
        if self.X is not None and preprocessor:
            preprocessor.apply(self)
            
        

    def get_test_set(self):
        """
        Returns the test set corresponding to this Flickr_Image_Toronto instance
        Generally speaking, we use Flickr unlabelled images for representation learning,
        labelled ones for validation or test
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['which_cat'] = 'labelled'
        args['including_label'] = False
        args['start'] = None
        args['stop'] = None
        return Flickr_Text_Toronto(**args)
