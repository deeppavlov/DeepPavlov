Hyperparameters optimization
============================

You can search for best hyperparameters of your model in DeepPavlov by means of cross-validation.

Cross-validation
~~~~~~~~~~~~~~~~

You can run cross-validation in DeepPavlov to select best parameters of your model.
For this purpose you have to run special command 'paramserach'. for example:

.. code:: bash

    python -m deeppavlov.paramsearch path_to_json_config.json --folds 5


Parameters
----------

Cross validation command have several parameters:

-  ``config_path``:
    Specify config path, where you model is located.
-  ``--folds``:
    This parameter shows how many folds you need in cross validation.
    Do you want to use leave one out cross validation instead of folds?
    Just specify this: ``--folds loo``.
    If you want not to cross-validate just omit this parameter.
-  ``--search_type``:
    This parameter is optional - default value is "grid" (grid search).


.. note::

    Folds will be created automatically from union of train and validation datasets.


Special parameters in config
----------------------------
Config file of model should be consist of parameters ranges for search.
For example, you try to optimize regularization coefficient in model,
so you should add additional parameter in config with suffix '_range'.
Let's see example for logistic regression model:

.. code:: python

      {
        "class_name": "faq_logreg_model",
        "in": "q_vect",
        "fit_on": ["q_vect", "y"],
        "c": {"search_choice": [1, 10, 100, 1000]},
        "out": ["answer", "score"]
      }

In this example parameter "c" described as search_choice, values for grid search:

.. code:: python

    {"search_choice": [value_0, ..., value_n]}


Results
-------
As a result you'll have new json config with best model parameters.
It'll be stored in the same directory as config file and will have suffix '_cvbest.json'.
Also you'll see final log messages about best model:

.. code:: bash

    INFO in '__main__'['paramsearch'] at line 169: Best model params: {'C': 10000, 'penalty': 'l1', 'accuracy': 0.81466}
    INFO in '__main__'['paramsearch'] at line 184: Best model saved in json-file: path_to_model_config_cvbest.json
