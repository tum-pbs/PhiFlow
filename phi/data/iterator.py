from .dataset import *
from .channel import *


class DataIterator(object):

    def __init__(self, dataset, channels):
        self._dataset = dataset
        self._channels = channels
        self._index = 0