from abc import ABCMeta, abstractmethod

class SiameseNetwork(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def load_initial_emb_matrix(self):
        pass

    @abstractmethod
    def train_on_batch(self, batch, y):
        pass

    @abstractmethod
    def predict_score_on_batch(self, batch):
        pass