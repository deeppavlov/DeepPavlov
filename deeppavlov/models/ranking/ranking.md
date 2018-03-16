[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Neural Model for Ranking

There is a task in NLP to retrieve the most suitable response to a given
context from some response database. A similar task is to find
the context closest semantically to a specific context from a
database with contexts. The code in this repository uses a deep learning
framework for the question answer selection task. Currently, a basic
model is implemented with bidirectional long short-term memory 
(biLSTM) and without attention. The model is applied to the InsuranceQA dataset
 https://github.com/shuzi/insuranceQA .
  
## Infer from pre-trained model

To use the pre-trained model for inference one should run the following command:
```
python deep.py interact configs/ranking/insurance_config.json
```
## Train model

To train the model on the InsuranceQA dataset one should run the command:
```
python deep.py train configs/ranking/insurance_config.json
```
All parameters for [insurance_config.json](../../configs/ranking/insurance_config.json) config file are presented in the table below.

#### Configuration parameters:  

|   Parameter         |  Description                                                      | 
|---------------------|-------------------------------------------------------------------|
| **dataset_reader**  | **reads datasets from files.** |
|   name              | a registered name of the dataset reader. |
|   data_path         | a directory where data files are stored. | 
| **dataset_iterator** | **provides models with data.** |
|   name              | a registered name of the dataset. | 
|   seed              | int, seed for the batch generator. |
|   sample_candiates  | {"global", "pool"}. The method of negative sampling in train data. If "pool", negative candidates for each data sample should be provided. If "global", negative sampling over the whole data is performed.|
|   sample_candiates_valid  | {"global", "pool"}. The method of selecting_candidates for ranking in valid data. If "pool", candidates for ranking for each data sample should be provided. If "global",  all data samples are taken as candidates for ranking.|
|   sample_candiates_test  | {"global", "pool"}. The method of selecting_candidates for ranking in valid data. If "pool", candidates for ranking for each data sample should be provided. If "global",  all data samples are taken as candidates for ranking.|
| num_negative_samples | int, the number of negative samples to use if "sample_candiates" is set to "pool". | 
| num_ranking_samples_valid |  int, the number of negative samples to use if "sample_candiates_valid" is set to "pool". |
| num_ranking_samples_test |  int, the number of negative samples to use if "sample_candiates_test" is set to "pool".  |
 **chainer**         | **pipeline from heterogeneous components.**     |
| in                  | a user-defined list of input names, i.e ["x"], ["x0", "x1"]. |
| in_y                | a user-defined list of input target names , i.e ["y"], ["y0", "y1"]. |
| out                 | a user-defined list of output names, i.e. ["y_pred"], ["y_pred0", "y_pred1"]. |
| **pipe**            | **contains the sequence of model components (including vocabs, preprocessors, main components, postprocessors etc.).**|
|                     | **parameters of the main part of a model** |
| in                  | the same as "in" parameter in "chainer". |
| in_y                | the same as "in_y" parameter in "chainer". |
| out                 | the same as "out" parameter in "chainer". |
| name                | a registered name of the model.  | 
| device_num | int, GPU card number to train the model, if several cards are available in the system. |
| load_path           | a path to a file from which model files will be loaded.    |
| save_path           | a path to a file where model files will be saved.   |
| train_now | bool, if it is True, than the model training will be done, else validation and test only.
| vocabs_path | a path to a directory with data files from where the model vocabularies will be built. |
| download_url | url where a pretrained model with word embeddings is stored.|
| seed | seed to initialize the model weights |

| dense_size          | size of dense layer that is followed by the classification dense layer    | 
| lear_rate           | learning rate for training    | 
| lear_rate_decay     | learning rate decay for training          | 
| optimizer           | optimizer for training    <br />*SetOfValues*: any method from keras.optimizers                         |
| loss                | loss for training       <br />*SetOfValues*: any method from keras.losses                             |
| **train**           | **parameters for training** |
| epochs              | number of epochs for training    |
| batch_size          | batch size for training    |
| metrics             | metrics to be used for training  <br />*SetOfValues*: any method from from keras.metrics         | 
| validation_patience | parameter of early stopping: for how many epochs the training can continue without improvement of metric value on the validation set           | 
| val_every_n_epochs  | frequency of validation during training (validate every n epochs)       | 
| val_every_n_batches | frequency of validation during training (validate every n batches)      | 
| show_examples       | whether to print training information or not        | 
