from abc import ABCMeta, abstractmethod

class SiameseEmbeddingsNetwork(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def embeddings_model(self):
        pass