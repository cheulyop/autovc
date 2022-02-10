from abc import ABCMeta, abstractmethod


class Pipeline(metaclass=ABCMeta):
    @abstractmethod
    def run(self, config):
        pass


class Metric(metaclass=ABCMeta):
    @abstractmethod
    def compute(self, results):
        pass
