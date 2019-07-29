Register your model
===================

In order to extend the library, you need to register your classes and functions; it is done in two steps.

1. Decorate your :class:`~deeppavlov.core.models.component.Component`
   (or :class:`~deeppavlov.core.data.dataset_reader.DatasetReader`,
   or :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator`,
   or :class:`~deeppavlov.core.data.data_fitting_iterator.DataFittingIterator`)
   using :func:`~deeppavlov.core.common.registry.register` and/or metrics function
   using :func:`~deeppavlov.core.common.metrics_registry.register_metric`.

2. Rebuild the registry running from DeepPavlov root directory:

::

    python -m utils.prepare.registry

This script imports all the modules in deeppavlov package, builds the registry from them and writes it to a file.


However, it is possible to use some classes and functions inside configuration files without registering them explicitly.
There are two options available here:

- instead of ``{"class_name": "registered_component_name"}`` in config file use key-value pair similar to
  ``{"class_name": "my_package.my_module:MyClass"}``

- if your classes/functions are properly decorated but not included in the registry, use ``"metadata"`` section of
  your config file specifying imports as ``"metadata": {"imports": ["my_local_package.my_module", "global_package.module"]}``;
  then the second step described above will be unnecessary (local packages are imported from the current working
  directory).
