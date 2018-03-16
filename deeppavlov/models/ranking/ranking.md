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
| **dataset_reader**  | **an object that reads datasets from files** |
|   name              | registered name of the dataset reader|
|   data_path         | directory where data files are stored                          | 
| **dataset**         | **an object that provides models with data|
|   name              | registered name of the dataset        <br />*SetOfValues*:  "intent_dataset", classification_dataset"     | 
|   seed              | seed for the batch generator              |
|   sample_candiates  | {"global", "pool"}. The method of negative sampling in train data. If "pool", negative samples for each data sample should be provided. If "global", negative sampling over the whole data is performed.|
|   sample_candiates_valid  | {"global", "pool"}. The method of selecting_candidates for ranking in valid data. If "pool", candidates for ranking for each data sample should be provided. If "global",  all data samples are taken as candidates for ranking.|
|   sample_candiates_test  | {"global", "pool"}. The method of selecting_candidates for ranking in valid data. If "pool", candidates for ranking for each data sample should be provided. If "global",  all data samples are taken as candidates for ranking.|

 **chainer**         | **chainer is a structure that receives tuples `(in, in_y)` and produces `out`**     |
| in                  | user-defined name of input (or list of names in case of multiple inputs) <br />*SetOfValues*: list of names, i.e ["x"], ["x0", "x1"] |
| in_y                | user-defined name of input targets (or list of names in case of multiple input targets) <br />*SetOfValues*: list of names, i.e ["y"], ["y0", "y1"] |
| out                 | user-defined name of output (or list of names in case of multiple outputs) <br />*SetOfValues*: list of names, i.e ["y_pred"], ["y_pred0", "y_pred1"] |
| **pipe**            | **list that contains the sequence of model components (including vocabs, preprocessors, postprocessors etc.)**|
|                     | **parameters of the vocabulary**  |
| id                  | name of the considered model for further references |
| name                | registered name of the vocab    <br />*SetOfValues*: "default_vocab"  | 
|  fit_on             | whether to create the vocab over x and/or y fields of dataset  <br />*SetOfValues*: list of names defined in chainer.in or chainer.in_y  |
|  level              | character-level or token-level tokenization      <br />*SetOfValues*: "char", "token"   |
| load_path           | path to file from which the vocab with classes will be loaded    |
| save_path           | path to file where vocab with classes will be saved    |
|                     | **parameters for building the main part of a model** |
| in                  | training samples to the model <br />*SetOfValues*: list of names from chainer.in, chainer.in_y or outputs of previous models |
| in_y                | target values for the training samples, compulsory for training <br />*SetOfValues*: list of names from chainer.in, chainer.in_y or outputs of previous models  |
| out                 | user-defined name of the output (or list of names in case of multiple outputs) <br />*SetOfValues*: list of names |
| main                | determines main part of the pipe |
| name                | registered name of model  | 
| load_path           | path to file from which model files will be loaded    |
| save_path           | path to file where model files will be saved    |
| classes             | list of class names. In this case they could be simply obtained from `classes_vocab.keys()` method |
| opt                 | **parameters for building a model** |
| model_name          | method of the class KerasIntentModel that corresponds to the model <br />*SetOfValues*: "cnn_model", "dcnn_model"   | 
| text_size           | length of each sample in words      | 
| confident_threshold | probability threshold for an instance belonging to a class  <br />*SetOfValues*: \[0., 1.\]                       | 
| kernel_sizes_cnn    | kernel sizes for shallow-and-wide and deep CNN models        | 
| filters_cnn         | number(-s) of filters for shallow-and-wide (deep) CNN   | 
| dense_size          | size of dense layer that is followed by the classification dense layer    | 
| lear_rate           | learning rate for training    | 
| lear_rate_decay     | learning rate decay for training          | 
| optimizer           | optimizer for training    <br />*SetOfValues*: any method from keras.optimizers                         |
| loss                | loss for training       <br />*SetOfValues*: any method from keras.losses                             |
| coef_reg_cnn        | coefficient for kernel l2-regularizer for convolutional layers   |
| coef_reg_den        | coefficient for kernel l2-regularizer for dense layers  |
| dropout_rate        | dropout rate for training    |
| **embedder**        | **object that produces word vectors** |
| embedder.name       | registered name of embedder  <br />*SetOfValues*:"fasttext"   |
| embedder.load_path  | path to file from which the embedding binary file will be loaded    |
| embedder.emb_module | fasttext library to use  <br />*SetOfValues*: "fasttext", "pyfasttext", "gensim"            | 
| embedder.dim        | dimension of embeddings    | 
| **tokenizer**       | **object that tokenizes text** |
| tokenizer.name      | registered name of tokenizer <br />*SetOfValues*: "nltk_tokenizer"                              | 
| tokenizer.tokenizer | tokenizer from nltk.tokenize to use  <br />*SetOfValues*:  any method from nltk.tokenize    |  
| **train**           | **parameters for training** |
| epochs              | number of epochs for training    |
| batch_size          | batch size for training    |
| metrics             | metrics to be used for training  <br />*SetOfValues*: any method from from keras.metrics         | 
| validation_patience | parameter of early stopping: for how many epochs the training can continue without improvement of metric value on the validation set           | 
| val_every_n_epochs  | frequency of validation during training (validate every n epochs)       | 
| val_every_n_batches | frequency of validation during training (validate every n batches)      | 
| show_examples       | whether to print training information or not        | 
