Parameters evolution for DeepPavlov models
==========================================

This repository contains implementation of parameters evolution for
DeepPavlov models.

Evolution process can be described in the following way:

-  Initialize parameters of evolutionary process setting the following
   arguments to ``evolve.py``:
-  ``--p_size`` - number of individuals (models) per population
   (*Default: 10*).
-  ``--key_main_model`` - key of the dictionary in config containing the
   model being trained (see description below) (Default: "main").
-  ``--p_cross`` - probability of crossover for a parent pair (*Default:
   0.2*).
-  ``--pow_cross`` - crossover power - portion of evolving parameters
   that will be exchanged between parents during crossover (Default:
   0.1).
-  ``--p_mut`` - probability of mutation for a parameter (*Default:
   1.*).
-  ``--pow_mut`` - mutation power - maximal portion of maximal possible
   value of parameter which can be added or subtracted during mutation
   (Default: 0.1).
-  ``--gpus`` - available GPUs divided by comma "," (*Default: -1 means
   CPU support*). If one runs ``evolve.py`` with assigned
   ``CUDA_VISIBLE_DEVICES``, gpus are either ordinal numbers of device
   within those from ``CUDA_VISIBLE_DEVICES`` (e.g.
   ``CUDA_VISIBLE_DEVICES=3,4,5`` and ``--gpus 1,2`` mean running models
   on ``4,5`` original GPUs) or all devices from
   ``CUDA_VISIBLE_DEVICES`` if gpus is not given.
-  ``--train_partition`` - if train file is too big to train (recommeded
   to divide train files if train dataset is more than 100 thousands
   examples), one can split it in ``train_partition`` number of files,
   save it calling "any\_name\_{0}.any\_extension", ...,
   "any\_name\_{``train_partition``\ }.any\_extension". In
   dataset\_reader "train" field indicate the first one file. Population
   is trained on the N\_{population} % ``train_partition`` part of the
   dataset (*Default: 1*).
-  ``--start_from_population`` - the number of population to start from
   that is needed to restart population (*Default: 0 means starts from 0
   population*).
-  ``--path_to_population`` - path to the directory
   "population\_{``start_from_population``\ }". Should be given if
   ``start_from_population`` is not 0 (*Default: ""*).
-  ``--elitism_with_weights`` - whether to initialize elite models with
   pre-trained weights from previous population or not (*Default: False
   means save elite models without weights. If parameter is given, then
   save elite models with weights*).
-  ``--iterations`` - number of iterations to conduct (*Default: -1
   means infinite number of iterations (while loop)*).

-  **Warning**: ``metrics`` can not be evolved because the main metric
   determines evolutionary process.

-  Current version allows to evolve any parameter of the config that is
   an item of some dictionary in config file. One can make a copy of a
   usual DeepPavlov model config, and reassign parameters that can be
   tuned during evolution.
   To evolve some parameter one has to assign it to a dictionary of one
   of the following type:
-  ``{"evolve_range": [min_value, max_value]}`` - values uniformly
   distributed on the given interval,
-  ``{"evolve_range": [min_value, max_value], "scale": "log"}`` - values
   distributed on the given interval logariphmically,
-  ``{"evolve_range": [min_value, max_value], "discrete": true}`` -
   discrete values uniformly distributed on the following interval,
-  ``{"evolve_bool": true}`` - bool values,
-  ``{"evolve_choice": true, "values": [value_0, ..., value_n]}`` -
   values uniformly taking on of the given values.

-  Choose the main model in the pipe being evolved. Find or add extra
   parameter that determines this model (for example, existing
   ``"main": true``). The dictionary - model containing this parameter
   as a key will be trained (do not forget to give this parameter's name
   to ``key_main_model``). Change ``save_path`` and ``load_path`` of
   this model to any ABSOLUTE paths (VERY IMPORTANT) to folder where
   population will be saved.

-  All the models in pipe that contain key ``fit_on`` will be trained
   every time separately for each model and saved to the same directory
   with model and called ``fitted_model_{i}``.

That's all you need to change in the config. Now let's mode on to the
example.

Example
-------

-  If one prefers to run evolution on some provided by DeepPavlov
   dataset,
   firstly, download embeddings and datasets.
   Consider parameters evolution on SNIPS dataset, download data
   running the following command providing
   corresponding name of the config file:

   ::

       cd deeppavlov
       python deep.py download configs/intents/intents_snips.json

-  To evolve the model run the following command providing corresponding
   name of the config file (see above):

   ::

       cd deeppavlov
       python evolve.py configs/evolution/evolve_intents_snips.json

-  Folder ``download/evolution/classification/intents_snips`` will be
   created. Each population will be saved in a folder
   ``download/evolution/classification/intents_snips/population_i`` each
   of which contains ``population_size`` folders ``model_i`` consisting
   of saved model files explicitly, saved files of models from pipe that
   has a key "fit\_on", ``out.txt`` and ``err.txt`` with logs of
   ``deep.py train`` script from training each model separately, and
   ``config.json`` with config for this individual.

-  Now one can open iPython Notebook file
   ``deeppavlov/models/evolution/Results_analysis.ipynb``, set
   ``CONFIG_FILE`` to config file path and run cells to see evolution
   results.
