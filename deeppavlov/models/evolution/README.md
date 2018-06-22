[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Parameters evolution for DeepPavlov models

This repository contains implementation of parameters evolution for DeepPavlov models.

Evolution process can be described in the following way:
* Initialize parameters of evolutionary process:
  - `p_size` - number of individuums (models) per population
  - `key_main_model` - key of the dictionary in config containing the model being trained.
  - `p_cross` - probability of crossover for a parent pair
  - `pow_cross` - crossover power - portion of evolving parameters that will be exchanged between parents during crossover
  - `p_mut` - probability of mutation for a parameter
  - `pow_mut` - mutation power - maximal portion of maximal possible value of parameter which can be added or subtracted during mutation
  - `gpus` - available GPUs divided by comma "," (default "-1" means CPU support; "0,3,5,2" means visible 0, 2, 3, 5 GPUs)
  - `train_partition` - if train file is too big to train (recommeded to divide train files if train dataset is more than 100 thousands examples), one can split it in `train_partition` number of files, save it calling "any_name_{0}.any_extension", ..., "any_name_{`train_partition`}.any_extension". In dataset_reader "train" field indicate the first one file. Population is trained on the N_{population} % `train_partition` part of the dataset.
  - `start_from_population` - the number of population to start from that is needed to restart population, for example (by feault, starts from 0 population).
  - `path_to_population` - path to the directory "population_{`start_from_population`}". Should be given if `start_from_population` is not 0.
  - `elitism_with_weights` - binary value (set of values: "0", "1") - whether to initialize elite models with pre-trained weights from previous population or not

## Example 

If one prefers to run evolution on some provided by DeepPavlov dataset,
firstly, download embeddings and datasets running the following command providing
corresponding name of the config file (see above):

```
cd deeppavlov
python deep.py download configs/intents/intents_snips.json
```

To evolve model of interest run the following command providing corresponding name of the config file (see above):
```
cd deeppavlov
python evolve.py interact configs/evolution/evolve_intents_snips.json
```

