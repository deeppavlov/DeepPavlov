[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Parameters evolution for DeepPavlov models

This repository contains implementation of parameters evolution for DeepPavlov models.

Evolution process can be described in the following way:
* Initialize parameters of evolutionary process:
  - `p_size` - number of individuals (models) per population
  - `key_main_model` - key of the dictionary in config containing the model being trained (see description below).
  - `p_cross` - probability of crossover for a parent pair
  - `pow_cross` - crossover power - portion of evolving parameters that will be exchanged between parents during crossover
  - `p_mut` - probability of mutation for a parameter
  - `pow_mut` - mutation power - maximal portion of maximal possible value of parameter which can be added or subtracted during mutation
  - `gpus` - available GPUs divided by comma "," (default "-1" means CPU support; "0,3,5,2" means visible 0, 2, 3, 5 GPUs)
  - `train_partition` - if train file is too big to train (recommeded to divide train files if train dataset is more than 100 thousands examples), one can split it in `train_partition` number of files, save it calling "any_name_{0}.any_extension", ..., "any_name_{`train_partition`}.any_extension". In dataset_reader "train" field indicate the first one file. Population is trained on the N_{population} % `train_partition` part of the dataset.
  - `start_from_population` - the number of population to start from that is needed to restart population, for example (by feault, starts from 0 population).
  - `path_to_population` - path to the directory "population_{`start_from_population`}". Should be given if `start_from_population` is not 0.
  - `elitism_with_weights` - binary value (set of values: "0", "1") - whether to initialize elite models with pre-trained weights from previous population or not

* Current version allows to evolve any parameter of the config that is an item of some dictionary in config file. One can make a copy of a usual DeepPavlov model config, and reassign parameters that can be tuned during evolution.
To evolve some parameter one has to assign it to a dictionary of one of the following type:
  - ```{"evolve_range": [min_value, max_value]}``` - values uniformly distributed on the given interval,
  - ```{"evolve_range": [min_value, max_value], "scale": "log"}``` - values distributed on the given interval logariphmically,
  - ```{"evolve_range": [min_value, max_value], "discrete": true}``` - discrete values uniformly distributed on the following interval,
  - ```{"evolve_bool": true}``` - bool values,
  - ```{"evolve_choice": true, "values": [value_0, ..., value_n]}``` - values uniformly taking on of the given values.

* Choose the main model in the pipe being evolved. Find or add extra parameter that determines this model (for example, existing `"main": true`). The dictionary - model containing this parameter as a key will be trained (do not forget to give this parameter's name to `key_main_model`). Change `save_path` and `load_path` of this model to any ABSOLUTE paths (VERY IMPORTANT) to folder where population will be saved.

* All the models in pipe that contain key `fit_on` will be trained every time separately for each model and saved to the same directory with model and called `fitted_model_{i}`.

That's all you need to change in the config. Now let's mode on to the example.

## Example 

* If one prefers to run evolution on some provided by DeepPavlov dataset,
firstly, download embeddings and datasets.

Consider parameters evolution on SNIPS dataset, download data running the following command providing
corresponding name of the config file:
```
cd deeppavlov
python deep.py download configs/intents/intents_snips.json
```
* To evolve the model run the following command providing corresponding name of the config file (see above):
```
cd deeppavlov
python evolve.py configs/evolution/evolve_intents_snips.json
```
* Folder `download/evolution/classification/intents_snips` will be created. Each population will be saved in a folder `download/evolution/classification/intents_snips/population_i` each of which contains `population_size` folders `model_i` consisting of saved model files explicitly, saved files of models from pipe that has a key "fit_on", `out.txt` and `err.txt` with logs of `deep.py train` script from training each model separately, and `config.json` with config for this individual.
