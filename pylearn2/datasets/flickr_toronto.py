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

import numpy
from pylearn2.datasets.flickr_image_toronto import Flickr_Image_Toronto
from pylearn2.datasets.flickr_text_toronto import Flickr_Text_Toronto
from theano.compat.six.moves import xrange
from pylearn2.datasets import control, cache
from pylearn2.datasets.exc import NotInstalledError
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng


class Flickr_Toronto(Flickr_Image_Toronto,Flickr_Text_Toronto):
    """
    Binarized, unlabeled version of the MNIST dataset

    Parameters
    ----------
    which_set : str
        Which subset of the dataset. Must be in ['train', 'valid', 'test']
    shuffle : bool, optional
        Whether to shuffle the dataset. Defaults to `False`.
    start : int, optional
        Defaults to `None`. If set, excludes examples whose index is inferior
        to `start`.
    stop : int, optional
        Defaults to `None`. If set, excludes examples whose index is superior
        to `stop`.
    axes : permutation of ['b', 0, 1, 'c'], optional
        Desired axes ordering of the topological view of the data. Defaults to
        ['b', 0, 1, 'c'].
    preprocessor : pylearn2.datasets.preprocessing.Preprocessor, optional
        Preprocessing to apply to the data. Defaults to `None`.
    fit_preprocessor : bool, optional
        Whether to fit the preprocessor to the data
    fit_test_preprocessor : bool, optional
        Whether to fit the preprocessor to the test data
    """

    def __init__(self, which_cat, which_usage, shuffle=False,
                 start=None, stop=None, axes=['b', 0, 1, 'c'],
                 preprocessor=None, fit_preprocessor=False,
                 fit_test_preprocessor=False):
        self.args = locals()
        cat_options = {'image': ['labelled',  'labelled_nnz',  'labelled_z',  'unlabelled',  'unlabelled_nnz',  'unlabelled_z'],
                       'text':['text_all_2000_labelled.npz', 'text_nnz_2000_labelled.npz', 'text_nnz_2000_unlabelled.npz', 'text_nnz_2000_labelled.npz',],
                       'joint': ''}
        usage_options = ['train', 'validation', 'test']
        if which_cat not in cat_options:
            raise ValueError('Unrecognized which_cat value "%s".' %
                             (which_cat,) + '". Valid values are ' +
                             '["image", "text", "joint"].')

        if which_cat != 'joint':
            path = "${PYLEARN2_DATA_PATH}/flickr/" + which_cat
                   
            im_path = serial.preprocess(path)

            # Locally cache the files before reading them
            datasetCache = cache.datasetCache
            im_path = datasetCache.cache_file(im_path)

            try:
                X = serial.load(im_path)
            except IOError:
                raise NotInstalledError("BinarizedMNIST data files cannot be "
                                        "found in ${PYLEARN2_DATA_PATH}. Run "
                                        "pylearn2/scripts/datasets/"
                                        "download_binarized_mnist.py to get "
                                        "the data")

        m, d = X.shape
        assert d == 28 ** 2
        if which_set == 'train':
            assert m == 50000
        else:
            assert m == 10000

        if shuffle:
            self.shuffle_rng = make_np_rng(None, [1, 2, 3],
                                           which_method="shuffle")
            for i in xrange(X.shape[0]):
                j = self.shuffle_rng.randint(m)
                # Copy ensures that memory is not aliased.
                tmp = X[i, :].copy()
                X[i, :] = X[j, :]
                X[j, :] = tmp

        super(BinarizedMNIST, self).__init__(
            X=X,
            view_converter=DefaultViewConverter(shape=(28, 28, 1))
        )

        assert not numpy.any(numpy.isnan(self.X))

        if start is not None:
            assert start >= 0
            if stop > self.X.shape[0]:
                raise ValueError('stop=' + str(stop) + '>' +
                                 'm=' + str(self.X.shape[0]))
            assert stop > start
            self.X = self.X[start:stop, :]
            if self.X.shape[0] != stop - start:
                raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                                 % (self.X.shape[0], start, stop))

        if which_set == 'test':
            assert fit_test_preprocessor is None or \
                (fit_preprocessor == fit_test_preprocessor)

        if self.X is not None and preprocessor:
            preprocessor.apply(self, fit_preprocessor)

    def adjust_for_viewer(self, X):
        """
        Adjusts the data to be compatible with a viewer that expects values to
        be in [-1, 1].

        Parameters
        ----------
        X : numpy.ndarray
            Data
        """
        return numpy.clip(X * 2. - 1., -1., 1.)

    def adjust_to_be_viewed_with(self, X, other, per_example=False):
        """
        Adjusts the data to be compatible with a viewer that expects values to
        be in [-1, 1].

        Parameters
        ----------
        X : numpy.ndarray
            Data
        other : WRITEME
        per_example : WRITEME
        """
        return self.adjust_for_viewer(X)

    def get_test_set(self):
        """
        Returns the test set corresponding to this BinarizedMNIST instance
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'test'
        args['start'] = None
        args['stop'] = None
        args['fit_preprocessor'] = args['fit_test_preprocessor']
        args['fit_test_preprocessor'] = None
        return BinarizedMNIST(**args)
