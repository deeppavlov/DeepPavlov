Choose the Framework
====================

DeepPavlov is built on top of the machine learning frameworks
`TensorFlow <https://www.tensorflow.org/>`__,
`Keras <https://keras.io/>`__ and `PyTorch <https://www.pytorch.org/>`__:

* BERT-based models on TensorFlow and PyTorch;
* Text classification on Keras and PyTorch;
* Text ranking and morpho-tagging on Keras;
* All other models on TensorFlow.

First, follow the instructions on :doc:`Installation page </intro/installation>`
to install the ``deeppavlov`` package for Python 3.6/3.7.

Depending on the considered NLP task, you need to choose one of the available frameworks.
The full list of available models is :doc:`here </features/overview>`.

- To install the requirements for the considered model, you can find the config file with the same configuration
  in terms of used components, and install the requirements in the following way:

        .. code:: bash

            python -m deeppavlov install -d <config_path>

        where ``<config_path>`` is path to the chosen model's config file (e.g. ``deeppavlov/configs/ner/slotfill_dstc2.json``)
        or just the file name without the `.json` extension (e.g. ``slotfill_dstc2``);
        ``-d`` downloads required data -- pretrained model files and embeddings (optional).

Trainer
-------

If you are going to use models on Keras or TensorFlow, in ``config["train"]``, you need to set ``"class_name": "nn_trainer"``;
If using PyTorch, you need to use ``"class_name": "torch_trainer"``, which differs from ``nn_trainer``
only in assigning ``torch.nn.Module.train()`` and ``torch.nn.Module.eval()`` models for PyTorch modules.


Text Classification on Keras or PyTorch
---------------------------------------

If you want to build your own architecture for **text classification** tasks, do the following in **Keras** or in **PyTorch**:

    .. code:: python

        # Keras
        from deeppavlov.models.classifiers.keras_classification_model import KerasClassificationModel
        # PyTorch
        # from deeppavlov.models.classifiers.torch_classification_model import TorchTextClassificationModel

        # Keras
        class MyModel(KerasClassificationModel):
        # Torch
        # class MyModel(TorchTextClassificationModel):

            def my_network_architecture(self, **kwargs):
                model = <create Keras/Torch model using parameters from kwargs>
                return model

    In the config file, assign ``"class_name": "module.path.to.my.model.file:MyModel"`` 
    and ``"model_name": "my_network_architecture"``
    in the dictionary with the main model.
    Don't forget to set ``torch_trainer`` or ``nn_trainer`` (for PyTorch) or ``nn_trainer`` (for TensorFlow and Keras).

Other NLP-tasks on TensorFlow, Keras, or PyTorch
------------------------------------------------

- If you want to build your own model for **some other NLP** task, do the following in **Keras** or **PyTorch**:

    .. code:: python

        # Keras
        from deeppavlov.core.models.keras_model import LRScheduledKerasModel
        # PyTorch
        # from deeppavlov.core.models.torch_model import TorchModel

        # Keras
        class MyModel(LRScheduledKerasModel):
        # Torch
        # class MyModel(TorchModel):

            def train_on_batch(x, y, *args, **kwargs):
                <your code here>
                return loss

            def __call__(data, *args, **kwargs):
                <your code here>
                return predictions

            def my_network_architecture(self, **kwargs):
                model = <create Keras/Torch model using parameters from kwargs>
                return model

    In the config file, assign ``"class_name": "module.path.to.my.model.file:MyModel"`` 
    and ``"model_name": "my_network_architecture"``
    in the dictionary with the main model.
    Don't forget to set ``torch_trainer`` (for PyTorch) or ``nn_trainer`` (for TensorFlow and Keras).


- If you want to build your own model for **some other NLP** task, do the following in **TensorFlow**:

    .. code:: python

        from deeppavlov.core.models.tf_model import LRScheduledTFModel

        class MyModel(LRScheduledTFModel):

            def _init_graph(self):
                <your code here>

            def _init_placeholders(self):
                <your code here>

            def _init_optimizer(self):
                <your code here>

            def _build_feed_dict(self, *variables):
                <your code here>
                return feed_dict

            def train_on_batch(x, y, *args, **kwargs):
                <your code here>
                feed_dict = self._build_feed_dict(*variables)
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                return {"loss": loss}

            def __call__(data, *args, **kwargs):
                <your code here>
                feed_dict = self._build_feed_dict(*variables)
                predictions = self.sess.run([self.predictions], feed_dict=feed_dict)
                return predictions.tolist()

    In the config file, assign ``"class_name": "module.path.to.my.model.file:MyModel"`` 
    and ``"model_name": "my_network_architecture"``
    in the dictionary with the main model; Also, set all the necessary parameters in the same dictionary.
    Don't forget to set  ``nn_trainer`` (for TensorFlow).
