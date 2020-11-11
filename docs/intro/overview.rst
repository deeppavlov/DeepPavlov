Conceptual overview
===================

Our goal is to enable AI-application developers and researchers with:

-  A set of pre-trained NLP models, pre-defined dialog system components
   (ML/DL/Rule-based), and pipeline templates;
-  A framework for implementing and testing their own dialog models;
-  Tools for application integration with adjacent infrastructure
   (messengers, helpdesk software, etc.);
-  Benchmarking environments for conversational models and uniform access
   to relevant datasets.

.. image:: ../_static/dp_agnt_diag.png


Key Concepts
------------

-  A ``Skill`` fulfills the user’s goal in some domain. Typically, this is
   accomplished by presenting information or completing transaction
   (e.g. answer question by FAQ, booking tickets, etc.). However, for
   some tasks, a successful interaction is defined as continuous
   engagement (e.g. chit-chat).
-  A ``Model`` is any NLP model that doesn't necessarily communicates
   with the user in natural language.
-  A ``Component`` is a reusable functional part of a ``Model`` or ``Skill``.
-  ``Rule-based Models`` cannot be trained.
-  ``Machine Learning Models`` can be trained only stand alone.
-  ``Deep Learning Models`` can be trained independently and in an
   end-to-end mode being joined in a chain.
-  A ``Skill Manager`` performs selection of the ``Skill`` to generate the
   response.
-  A ``Chainer`` builds a model pipeline from heterogeneous
   components (Rule-based/ML/DL). It allows one to train and infer models in
   a pipeline as a whole.

The smallest building block of the library is a ``Component``.
A ``Component`` stands for any kind of function in an NLP pipeline. It can
be implemented as a neural network, a non-neural ML model, or a
rule-based system.

``Component``\ s can be joined into a ``Model`` or a ``Skill``. A ``Model``
solves a larger NLP task than a ``Component``. However, in terms of
implementation, ``Model``\ s are not different from ``Component``\ s. 
What differentiates a ``Skill`` from a ``Model`` is that 
the input and output of a ``Skill`` should both be strings.
Therefore, ``Skill``\ s are usually associated with
dialogue tasks.

DeepPavlov is built on top of the machine learning frameworks
`TensorFlow <https://www.tensorflow.org/>`__,
`Keras <https://keras.io/>`__ and `PyTorch <https://www.pytorch.org/>`__. Other external libraries can be used to
build basic components.

