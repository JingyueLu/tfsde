import abc
from tensorflow.keras import layers


class InterpolationBase(layers.Layer, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def grid_points(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def interval(self):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, t):
        raise NotImplementedError

    @abc.abstractmethod
    def derivative(self, t):
        raise NotImplementedError
