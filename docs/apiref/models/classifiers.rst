deeppavlov.models.classifiers
=============================
Component for classification tasks (intents, sentiment, etc) on word-level. Shallow-and-wide CNN, Deep CNN, BiLSTM,
BiLSTM with self-attention and other models are presented. The model allows multilabel classification of sentences.

.. automodule:: deeppavlov.models.classifiers
   :members:

.. autoclass:: deeppavlov.models.classifiers.keras_classification_model.KerasClassificationModel

   .. automethod:: cnn_model
   .. automethod:: dcnn_model
   .. automethod:: cnn_model_max_and_aver_pool
   .. automethod:: bilstm_model
   .. automethod:: bilstm_bilstm_model
   .. automethod:: bilstm_cnn_model
   .. automethod:: cnn_bilstm_model
   .. automethod:: bilstm_self_add_attention_model
   .. automethod:: bilstm_self_mult_attention_model
   .. automethod:: bigru_model
