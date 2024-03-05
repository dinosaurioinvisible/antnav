import numpy as np
from abc import ABC, abstractmethod
from source.utils import pick_im_matcher

class Navigator():
    
    def __init__(self, route_images, matcher='mae', deg_range=(-180, 180), degree_shift=1, **kwargs) -> None:
        self.route_end = len(route_images)
        self.route_images = route_images
        self.deg_range = deg_range
        self.deg_step = degree_shift
        self.degrees = np.arange(*deg_range)
        self.matcher = pick_im_matcher(matcher)
        self.argminmax = np.argmin
    
    @abstractmethod
    def get_heading():
        pass

    @abstractmethod
    def navigate():
        pass

    @abstractmethod
    def get_rec_headings(self):
        pass
    
    @abstractmethod
    def get_index_log(self):
        pass

    @abstractmethod
    def get_rsims_log(self):
        pass
    
    @abstractmethod
    def get_best_sims(self):
        pass
    
    @abstractmethod
    def get_name(self):
        pass

