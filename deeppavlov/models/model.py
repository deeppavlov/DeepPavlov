"""This is a base abstract class for other abstract model classes. Your model should be inherited
from one of the particluar classes, :class:`TensorflowModel`, :class:`SklearnModel`,
:class:`NonTrainableModel` """

from deeppavlov.common.attributes import abstract_attribute


class Model:

    vocab = None

    def infer(self, data, *inputs):
        """
        Predict data.
        :param data: any type of input data
        :return: any type of predicted values
        """
        # TODO add 'Abstract Method' to all NotImplementedError errors
        # TODO check if @abstractmethod decorator is available here.
        raise NotImplementedError
