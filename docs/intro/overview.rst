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

-  A ``Model`` is any NLP model that doesn't necessarily communicates
   with the user in natural language.
-  A ``Component`` is a reusable functional part of a ``Model``.
-  ``Rule-based Models`` cannot be trained.
-  ``Machine Learning Models`` can be trained only stand alone.
-  ``Deep Learning Models`` can be trained independently and in an
   end-to-end mode being joined in a chain.
-  A ``Chainer`` builds a model pipeline from heterogeneous
   components (Rule-based/ML/DL). It allows one to train and infer models in
   a pipeline as a whole.

The smallest building block of the library is a ``Component``.
A ``Component`` stands for any kind of function in an NLP pipeline. It can
be implemented as a neural network, a non-neural ML model, or a
rule-based system.

``Component``\ s can be joined into a ``Model``. A ``Model``
solves a larger NLP task than a ``Component``. However, in terms of
implementation, ``Model``\ s are not different from ``Component``\ s.

Most of DeepPavlov models are built on top of `PyTorch <https://www.pytorch.org/>`__.
Other external libraries can be used to build basic components.
