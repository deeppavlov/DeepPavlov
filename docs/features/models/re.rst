Relation Extraction (RE)
==============================

Relation extraction is the task of detecting and classifying the relationship between to entities in text.
DeepPavlov provides the document-level relation extraction meaning that the relation can be detected between the entities that are not in one sentence.
Currently, RE is available only for English language.

RE Component Architecture
-----------------------

Train the model
-----------------------

The RE model can be trained using the following command:

.. code:: bash

    python3.6 -m deeppavlov train re_docred


Use the trained RE model
-----------------------

Currently, we provide RE trained on `DocRED<https://www.aclweb.org/anthology/P19-1074/>`__ corpus.
We based our model on the `Adaptive Thresholding and Localized Context Pooling<https://arxiv.org/pdf/2010.11304.pdf>`__ model and use NER entity tags as additional input.


The trained model weights can be loaded with the following command:

.. code:: bash

    python3.6 -m deeppavlov download re_docred

The trained model can be used for inference with the following code:

.. code:: python

    from deeppavlov import configs, build_model
    re = build_model(configs.relation_extraction.re_docred, download=False)

    sentence_tokens = ["Barack", "Obama", "is", "married", "to", "Michelle", "Obama", "," "born", "Michelle", "Robinson", "."]
    entity_info = [[(0, 2)], [(5, 7), (9, 11)], "PER", "PER"]
    re([[sentence_tokens, entity_info]])
    >> [["spouse"]]
