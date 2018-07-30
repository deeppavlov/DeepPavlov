Conceptual overview
===================

Our goal is to enable AI-application developers and researchers with:

-  set of pre-trained NLP models, pre-defined dialog system components
   (ML/DL/Rule-based) and pipeline templates;
-  a framework for implementing and testing their own dialog models;
-  tools for application integration with adjacent infrastructure
   (messengers, helpdesk software etc.);
-  benchmarking environment for conversational models and uniform access
   to relevant datasets.

.. image:: ../_static/dp_agnt_diag.png


Key Concepts
------------

-  ``Agent`` is a conversational agent communicating with users in
   natural language (text).
-  ``Skill`` fulfills user’s goal in some domain. Typically, this is
   accomplished by presenting information or completing transaction
   (e.g. answer question by FAQ, booking tickets etc.). However, for
   some tasks a success of interaction is defined as continuous
   engagement (e.g. chit-chat).
-  ``Component`` is a reusable functional part of ``Skill``.
-  ``Rule-based Models`` cannot be trained.
-  ``Machine Learning Models`` can be trained only stand alone.
-  ``Deep Learning Models`` can be trained independently and in an
   end-to-end mode being joined in a chain.
-  ``Skill Manager`` performs selection of the ``Skill`` to generate
   response.
-  ``Chainer`` builds an agent/component pipeline from heterogeneous
   components (Rule-based/ML/DL). It allows to train and infer models in
   a pipeline as a whole.

The smallest building block of the library is ``Component``.
``Component`` stands for any kind of function in an NLP pipeline. It can
be implemented as a neural network, a non-neural ML model or a
rule-based system. Besides that, ``Component`` can have nested
structure, i.e. a ``Component`` can include other ``Component`` s.

``Component`` s can be joined into a ``Skill``. ``Skill`` solves a
larger NLP task compared to ``Component``. However, in terms of
implementation ``Skill``\ s are not different from ``Component``\ s. The
only restriction of ``Skill``\ s is that their input and output should
both be strings. Therefore, ``Skill``\ s are usually associated with
dialogue tasks.

``Agent`` is supposed to be a multi-purpose dialogue system that
comprises several ``Skill``\ s and can switch between them. It can be a
dialogue system that contains a goal-oriented and chatbot skills and
chooses which one to use for generating the answer depending on user
input.

DeepPavlov is built on top of machine learning frameworks
`TensorFlow <https://www.tensorflow.org/>`__ and
`Keras <https://keras.io/>`__. Other external libraries can be used to
build basic components.
