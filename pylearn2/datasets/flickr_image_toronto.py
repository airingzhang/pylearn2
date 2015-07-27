'''
Created on Jul 7, 2015

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
from theano.compat.six.moves import xrange
from pylearn2.datasets.dense_design_matrix import (
    DenseDesignMatrix,
    DefaultViewConverter
)
from pylearn2.datasets import control, cache
from pylearn2.datasets.exc import NotInstalledError
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng
from pylearn2.datasets.preprocessing import Standardize


class Flickr_Image_Toronto(DenseDesignMatrix):
    """
    Flickr image part, to make it easier to use for multimodal learning,
    we divide Flick dataset into 3 parts: image, text, and join.
    One can also refer to Flickr_Text_Toronto  and Flickr_Join_Toronto

    Parameters
    ----------
    which_cat : str
        Which category of the image dataset. Must be in ['labelled', 'unlabelled']
        If None, which_sub should not be none and it means both two categories.
    which_sub: str
        Which subcategory of the image dataset. Must be in ['z', and 'nnz']
        If None, which_cat should not be none and it means both two subcategories.
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

    def __init__(self, which_cat=None, which_sub=None, including_label = False, shuffle=False,
                 using_statisfile=False, start=None, stop=None, preprocessor=None):
        self.args = locals()
        cat_options = ['labelled','unlabelled']
        sub_options = ['z', 'nnz']
        if which_cat is None and which_sub is None:
            raise ValueError('Both which_cat and which_sub is None, at least one should be set')
        
        if which_cat is None and which_cat not in cat_options:
            raise ValueError('Unrecognized which_cat value "%s".' %
                             (which_cat,) + '". Valid values are ' +
                             '["labelled", "unlabelled"].')
            
        if which_sub is not None and which_sub not in sub_options:
            raise ValueError('Unrecognized which_sub value "%s".' %
                             (which_sub,) + '". Valid values are ' +
                             '["z", "nnz"].') 
        if which_cat is not None:       
            temp_path = "${PYLEARN2_DATA_PATH}/flickr/image/" + which_cat 
            if which_sub is None:
                path = temp_path + '/'
            else:  
                path = temp_path + '_' + which_sub + '/'
        else:
            path = ["${PYLEARN2_DATA_PATH}/flickr/image/"+ cat + '_' + which_sub + '/' for cat in which_cat]
            
        m_total = 0
        datasets = []
        if type(path) == list:
            initialized = False
            for path_item in path:                       
                im_path = serial.preprocess(path_item)
                files = glob.glob(im_path+'*.npy') 
                datasetCache = cache.datasetCache
                # Locally cache the files before reading them
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
                        numpy.concatenate((X,temp),axis = 0)
                      
        else:
            im_path = serial.preprocess(path)
            files = glob.glob(im_path+'*.npy') 
            datasetCache = cache.datasetCache
            # Locally cache the files before reading them
            initialized = False
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
                    numpy.concatenate((X,temp),axis = 0)
                    
        if which_cat is None:
            assert m_total == (which_sub == 'nnz' and 871499 or 128501) 
        
        elif which_cat == 'unlabelled':
            if which_sub is None:
                assert m_total == 975000
            else:               
                assert m_total == (which_sub == 'nnz' and 851050 or 123950) 
        else:
            if which_sub is None:
                assert m_total == 25000
            else:
                assert m_total == (which_sub == 'nnz' and 20449 or 4551) 
       
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
            
            if which_cat is None:
                if which_sub == 'z':
                    indices =indices_labelled['z_indices']
                else:
                    indices =indices_labelled['nnz_indices']  
    
            if which_cat == 'labelled':
                if which_sub is None:
                    indices = numpy.concatenate((indices_labelled['z_indices'], indices_labelled['nnz_indices']))
                else:
                    indices = which_sub == 'z' and indices_labelled['z_indices'] or indices_labelled['nnz_indices']
                    
            y = labels[indices[:]]     
                
            assert not numpy.any(numpy.isnan(self.y))        
            super(Flickr_Image_Toronto, self).__init__(
                X=X, y = y 
            )
            
            if shuffle:
                self.shuffle_rng = make_np_rng(None, [1, 2, 3],
                                           which_method="shuffle")
                for i in xrange(X.shape[0]):
                    j = self.shuffle_rng.randint(m)
                    # Copy ensures that memory is not aliased. together with label 
                    tmp = X[i, :].copy()
                    X[i, :] = X[j, :]
                    X[j, :] = tmp
                    tmp = y[i, :].copy()
                    y[i, :] = y[j, :]
                    y[j, :] = tmp
      
        else: 
            super(Flickr_Image_Toronto, self).__init__(
                X=X   
            )
            if shuffle:
                self.shuffle_rng = make_np_rng(None, [1, 2, 3],
                                           which_method="shuffle")
                for i in xrange(X.shape[0]):
                    j = self.shuffle_rng.randint(m)
                    # Copy ensures that memory is not aliased.
                    tmp = X[i, :].copy()
                    X[i, :] = X[j, :]
                    X[j, :] = tmp
        assert not numpy.any(numpy.isnan(self.X))        
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
                
                
        """
        if standardize preprocessor is used here, one should consider about loading the statisfile in flickr folder
        which is calculated by all the datasets, including both the unlabelled and labelled ones.
        """        
        if self.X is not None:
            if preprocessor:
                preprocessor.apply(self, can_fit=True)
            elif using_statisfile:
                im_path = serial.preprocess("${PYLEARN2_DATA_PATH}/flickr/flickr_stats.npz")
                try:
                    global_statis = numpy.load(im_path)
                except IOError:
                    raise NotInstalledError("Flickr_stats file cannot be "
                                                    "found in " + im_path + ". Please check the directory")
                self.X = (self.X - global_statis['mean'])/(1e-6 + global_statis['std'])
            

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
        args['which_sub'] = None
        args['including_label'] = False
        args['start'] = None
        args['stop'] = None
        args['preprocessor'] = Standardize
        return Flickr_Image_Toronto(**args)
