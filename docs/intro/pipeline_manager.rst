Pipeline Manager
================

Introduction:
-------------

In the process of research, a situation often arises where a series of similar experiments is necessary.
For instance one could work on intent classification task. Therefore the multitude of embeddings and models could be used.
Usually it's not clear which combination of those will give a better score. As a result, one would want
to try and train the model with all embeddings, comparing the results of each combination.
The :class:`~deeppavlov.pipeline_manager.PipelineManager` is designed to automate experiments of this kind.

|alt text| **Diagram 1.**

The structure of the experiment and its parameters is described as a special json config. It is given to the

:class:`~deeppavlov.pipeline_manager.PipelineManager` class as an argument when it is initialized. Based on this data,

and using the DeepPavlov library's functionality, the class automatically runs all of the experiments described in the
config, saves their description and results, and finally builds a table with .xlsx format in which the experiments are
sorted in descending order by the target metric.

Experiments can be run both sequentially and in parallel. Also, the :class:`~deeppavlov.pipeline_manager.PipelineManager`
class functionality includes hyperparameters search for individual models. “Random search” and “grid search”
are available.

Running a large number of experiments especially with complex neural models may take a large amount of time.
To avoid errors during the experiment a special test was added to check the correctness of pipelines.
During the test, all pipelines are trained on a small part of the original dataset. All of the temporary files are saved
in the folder “~/tmp/” and after successfully passing the test the folder is automatically deleted.
If the test is failed, the “~/tmp/” folder with all its content remains for debugging.

Usage
-----

First you would need to install additional requirements:

.. code:: bash

    python -m deeppavlov install <path_to_config>

After you wrote your config file, you can run your experiment by running in terminal command:

.. code:: bash

    python -m deeppavlov pipeline_search <path_to_config> [-d]

The ``-d`` parameter downloads
   - data required to train your model (embeddings, etc.);
   - a pretrained model if available (provided not for all configs).

Also you can run :class:`~deeppavlov.pipeline_manager.PipelineManager` in code:

.. code:: python

    from deeppavlov.pipeline_manager import PipelineManager
    pipeman = PipelineManager("path to your config file or config dict")
    pipeman.run()

**Examples:**
For a quick start, you can run test experiment on Snips dataset by this command from bash:

.. code:: bash

    python -m deeppavlov pipeline_search deeppavlov/configs/pipeline_manager/ner_dstc2.json. -d

Config description for Pipeline Manager:
----------------------------------------

Description of the structure of the experiments and
:class:`~deeppavlov.pipeline_manager.PipelineManager` class attributes is also described by the config file. It's main
difference from the :doc:`config_description </intro/config_description>` is that elements of the chainer can be a list:

|alt text3| **Diagram 2.** Conceptual example of :class:`~deeppavlov.pipeline_manager.PipelineManager` config.

And during the work of the :class:`~deeppavlov.pipeline_manager.PipelineManager` it will launch all of the components
combinations separately.

.. note::

    **WARNING!:** All components listed in one list must accept the same data type and format as input and output it.
    In other words, all components within the list should be compatible with their closest neighbors. Otherwise, an
    error will occur.

It is also recommended to add the "component_name" key in the description of the parameters of all components of the ``Chainer``.
The "component_name" key value will be displayed in the result table. If the key is not defined the default values
("class_name", "ref", "model_class") will be used.

The :class:`~deeppavlov.pipeline_manager.PipelineManager` operation mode is defined by the parameters:
     - **exp_name**
     - **experiments_root**
     - **launch_name**
     - **do_test**
     - **search_type**
     - **sample_num**
     - **save_best**
     - **multiprocessing**
     - **num_workers**
     - **use_gpu**
     - **gpu_memory_fraction**

This parameters are defined in the config file under the key “pipeline_search”. Here is simplify example:

.. code:: python

    {
        "dataset_reader": {...},
        "dataset_iterator": {...},
        "chainer": {
            "in": ["x"],
            "in_y": ["y"],
            "pipe": [ ... ],
            "out": ["pred_labels"]
            },
        "train": {...},
        "metadata": {...},
        "enumerate": {
            "exp_name": "lin_clf",
            "experiments_root": "./download/experiments/",
            "launch_name": "random_launch,
            "do_test": false,
            "search_type": "random",
            "sample_num": 10,
            "save_best": true,
            "multiprocessing": true,
            "max_num_workers": 4,
            "use_gpu": false,
            "gpu_memory_fraction": 1.0
            }
    }

You can look at the full config file for Pipeline Manager here :config:`deeppavlov/configs/pipeline_manager/ner_dstc2.json <ner_dstc2.json>`.

Directories structure
---------------------

When you start to work with the :class:`~deeppavlov.pipeline_manager.PipelineManager` in the path specified
through the parameter **experiments_root**, the following structure is created:

- {**experiments_root**}/
    - **date**/
        - **exp_name**/**launch_name**/
            - checkpoints/
                - pipe_1/
                    - config.json
                    - out.txt
                    - [others checkpoints files]
                - pipe_2/
                - ...
            - **exp_name**.json
            - logs.jsonl
            - exp_data.csv  # creating in the end of algorithm
            - exp_data.xlsx  # creating in the end of algorithm

**Explanations:**
 - The file “exp_data.xlsx” is a results table and created at the end of the algorithm.
 - The file **exp_name**.json is the log of the whole experiment, it contains a description of all the running
   experiments, their results, the time of the experiment, etc. This file is created at the start of the algorithm, and
   updated throughout the entire algorithm.
 - The “checkpoints/” folder is created when the algorithm is launched, and getting updated throughout its operation.
 - The file “checkpoints/pipe_{x}/config.json” is the default DP configuration for the pipeline
   “pipe_{x}” with all the necessary dependencies. So if you want to run the model trained in “pipe_{x}” to be
   validated or inferenced, you do not need to write the config again, it will be enough to refer to this file.
 - The file “checkpoints/pipe_{x}/out.txt” contains the contents of the std.err and std.out
   streams received from the training “pipe_{x}”.

Hyperparameter search
---------------------

Let's take a look at the example of the config with random search of hyperparameters:

.. code:: python

    {
        "chainer": {
            "in": ["x"],
            "in_y": ["y"],
            "pipe": [
                [...],
                ... ,
                [
                    {
                     "in": ["x_vec"],
                     "out": ["y_pred_probas"],
                     "fit_on": ["x_vec", "y_ids"],
                     "class_name": "sklearn_component",
                     "C": {"random_range": [1, 1000], "discrete": true, "scale": "log"},
                     "fit_intercept": {"random_bool": true},
                     "class_weight": {"random_choice": [null, "balanced"]},
                     "solver": {"random_choice": ["lbfgs", "newton-cg"]}
                    }
                ]
            ],
            "out": ["y_pred_labels"]
        }
    }

As you can see from the example, in the dictionaries with the description of the search, there are different keys
[**random_bool**, **random_choice**, **random_range**] and they determine the effect of sampling.
In the case of **random_bool**, the attribute value is randomly taken as True or False (and no matter what the value
of this key is). In the case of **random_choice**, one of the elements of the presented list is randomly selected.
And with **random_range** a number from the specified range is sampled randomly.

For the latter case, additional parameters **discrete** and **scale** are provided. The discrete option is self-explanatory:
it makes the values is chosen range discrete. The second one takes values from [None, “log”], if the parameter is “log”,
then sampling will take place on a logarithmic scale, the default value is None. So that way in the example above
variable 'C' will take random value from the range [1, 1000] on a logarithmic scale.

**sample_num** parameter is describing the number of unique hyperparameters set that will be generated from the range above.
The pipe will be generated separately for each set.

Here is example of config with grid_search of hyperparameters:

.. code:: python

    {
        "chainer": {
            "in": ["x"],
            "in_y": ["y"],
            "pipe": [
                [...],
                ... ,
                [
                    {
                     "in": ["x_vec"],
                     "out": ["y_pred_probas"],
                     "fit_on": ["x_vec", "y_ids"],
                     "class_name": "sklearn_component",
                     "C": {"grid_search": [1, 10, 100, 1000]},
                     "fit_intercept": {"grid_search": [false, true]},
                     "class_weight": {"grid_search": [null, "balanced"]},
                     "solver": {"grid_search": ["lbfgs", "newton-cg"]}
                    }
                ]
            ],
            "out": ["y_pred_labels"]
        }
    }

.. |alt text| image:: ../_static/pipeline_manager/PM_basic.png
.. |alt text3| image:: ../_static/pipeline_manager/pm_config.png
