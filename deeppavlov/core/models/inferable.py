"""
:class:`deeppavlov.models.model.Inferable` is an abstract base class that expresses the interface
for all models that can work for inferring. The scope of all inferring models is larger than the scope
of trainable models. For example, encoders can be inferred, but can't be trained.
All inferring models should inherit from this class.
"""
from abc import abstractmethod

from .serializable import Serializable


class Inferable(Serializable):
    """
    :attr:`train_now` expresses a developer intent for whether a model as part of a pipeline
    should be trained in the current experiment run or not.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def infer(self, instance, *args, **kwargs):
        """
        Infer a model. Any model can infer other model and ask it to do something (predict, encode,
        etc. via this method)
        :param instance: pass data instance to an inferring model
        :param args: all needed params for inferring
        :return a result of inferring
        """
        pass
