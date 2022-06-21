Choose the Framework
====================

DeepPavlov is built on top of the machine learning framework `PyTorch <https://www.pytorch.org/>`__.

First, follow the instructions on :doc:`Installation page </intro/installation>`
to install the ``deeppavlov`` package for Python 3.6/3.7/3.8/3.9

Depending on the considered NLP task, you need to choose one of the available frameworks.
The full list of available models is :doc:`here </features/overview>`.

- To install the requirements for the considered model, you can find the config file with the same configuration
  in terms of used components, and install the requirements in the following way:

        .. code:: bash

            python -m deeppavlov install -d <config_path>

        where ``<config_path>`` is path to the chosen model's config file (e.g. ``deeppavlov/configs/classifiers/insults_kaggle_bert.json``)
        or just the file name without the `.json` extension (e.g. ``insults_kaggle_bert``);
        ``-d`` downloads required data -- pretrained model files and embeddings (optional).

Trainer
-------

If you are going to use models on PyTorch, in ``config["train"]``, you need to set ``"class_name": "torch_trainer"``;


Text Classification on PyTorch
---------------------------------------

If you want to build your own architecture for **text classification** tasks, do the following:

    .. code:: python

        from deeppavlov.models.classifiers.torch_classification_model import TorchTextClassificationModel

        class MyModel(TorchTextClassificationModel):

            def my_network_architecture(self, **kwargs):
                model = <create Torch model using parameters from kwargs>
                return model

    In the config file, assign ``"class_name": "module.path.to.my.model.file:MyModel"`` 
    and ``"model_name": "my_network_architecture"``
    in the dictionary with the main model.

Other NLP-tasks
---------------

- If you want to build your own **PyTorch**-based model for **some other NLP** task, do the following:

    .. code:: python

        from deeppavlov.core.models.torch_model import TorchModel

        class MyModel(TorchModel):

            def train_on_batch(x, y, *args, **kwargs):
                <your code here>
                return loss

            def __call__(data, *args, **kwargs):
                <your code here>
                return predictions

            def my_network_architecture(self, **kwargs):
                model = <create Torch model using parameters from kwargs>
                return model

    In the config file, assign ``"class_name": "module.path.to.my.model.file:MyModel"`` 
    and ``"model_name": "my_network_architecture"``
    in the dictionary with the main model.
