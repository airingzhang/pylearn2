'''
Created on Aug 3, 2015

@author: ningzhang
'''
from pylearn2.sandbox.cuda_convnet.base_acts import UnimplementedError
from pylearn2.space import CompositeSpace
from pylearn2.datasets.dataset import Dataset


class CompositeDataset(Dataset):
    """
    This is a wrapper to combine multiple datasets into one for multimodal learning
    """
    def __init__(self, components):
        if components is None or len(components) < 1:
            raise UnimplementedError("You should used simple dataset instead of CompositeDataset")
        self.components = components
        self.space = CompositeSpace([component.specs(0) for component in components])
        self.source = tuple([component.specs(1) for component in components])
        self.specs = (self.space, self.source)
        batch_sizes = [component.batch_size for component in components]
        if any(batch_size != batch_sizes[0] for batch_size in batch_sizes) :
            raise UnimplementedError("CompositeDataset requires a unified bacthsize along all its components")
        self.batch_size = batch_sizes[0] 
        
        data_sizes = [component.get_num_examples() for component in components]
        if any(batch_size != batch_sizes[0] for batch_size in batch_sizes) :
            raise UnimplementedError("CompositeDataset requires a unified data size along all its components")
        self.data_size = data_sizes[0] 
    
    def get_num_examples(self):
        return self.data_size
    
    def iterator(self):
        return [component.iterator() for component in self.components]
    
        
        