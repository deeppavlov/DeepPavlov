[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Neural Model for Ranking

There is a task in NLP to retrieve the context closest semantically or the most
suitable response to a given context from some context (response) database. 
The code in this repository uses a deep learning
approach to address the question answer selection task. Currently, a basic
model is implemented with bidirectional long short-term memory 
(biLSTM), with max-pooling and without attention. The model is applied to the InsuranceQA dataset
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
All parameters from [insurance_config.json](../../configs/ranking/insurance_config.json) config file are described in the table below.

#### Configuration parameters:  

|   Parameter         |  Description                                                      | 
|---------------------|-------------------------------------------------------------------|
| **dataset_reader**  | **reads datasets from files.** |
|   name              | str, a registered name of the dataset reader. |
|   data_path         | str, a directory where data files are stored. | 
| **dataset_iterator** | **provides models with data.** |
|   name              | str, a registered name of the dataset. | 
|   seed              | int, a seed for a batch generator. |
|   sample_candiates  | {"global", "pool"}, a method of negative sampling in train data. If "pool", negative candidates for each data sample should be provided. If "global", negative sampling over the whole data will be performed.|
|   sample_candiates_valid  | {"global", "pool"}, a method of selecting_candidates for ranking in valid data. If "pool", candidates for ranking for each data sample should be provided. If "global",  all data samples will be taken as candidates for ranking.|
|   sample_candiates_test  | {"global", "pool"}, a method of selecting_candidates for ranking in valid data. If "pool", candidates for ranking for each data sample should be provided. If "global",  all data samples will be taken as candidates for ranking.|
| num_negative_samples | int, a number of negative samples to use if "sample_candiates" is set to "pool". | 
| num_ranking_samples_valid |  int, a number of negative samples to use if "sample_candiates_valid" is set to "pool". |
| num_ranking_samples_test |  int, a number of negative samples to use if "sample_candiates_test" is set to "pool".  |
 **chainer**         | **pipeline from heterogeneous components.**     |
| in                  | list of str, a user-defined list of input names, i.e ["x"], ["x0", "x1"]. |
| in_y                | list of str, a user-defined list of input target names , i.e ["y"], ["y0", "y1"]. |
| out                 | list of str, a user-defined list of output names, i.e. ["y_pred"], ["y_pred0", "y_pred1"]. |
| **pipe**            | **contains the sequence of model components (including vocabs, preprocessors, main components, postprocessors etc.).**|
|                     | **parameters of the main part of a model** |
| in                  | the same as "in" parameter in the "chainer". |
| in_y                | the same as "in_y" parameter in the "chainer". |
| out                 | the same as "out" parameter in the "chainer". |
| name                | str, a registered name of the model. | 
| device_num | int, a GPU card number to train the model on, if several cards are available in a system. |
| load_path           | str, a path to a file from which model files will be loaded.    |
| save_path           | str, a path to a file where model files will be saved.   |
| train_now | bool, if it is True, than the model training will be done, else validation and test only.
| vocabs_path | str, a path to a directory with data files from where the model vocabularies will be built. |
| download_url | str, a URL where a pre-trained model with word embeddings is stored.|
| embeddings | {"wor2vec", "fasttext"}, a type of the pre-trained embeddings model.
| seed | int, a seed to initialize the model weights. |
| max_sequence_length | int, a maximum number of tokens in an input sequence. If the sequence is shorter than the "max_sequence_length" it will be padded with a default token, otherwise the sequence will be truncated.| 
| padding | {"pre", "post"}, pad either before or after each sequence if it is shorter than "max_sequence_length". |
| truncating | {"pre", "post"}, remove values from sequences larger than "max_sequence_length", either at the beginning or at the end of the sequences. |
| reccurent | {"lstm", "bilstm"}, a type of a reccurent neural network (LSTM or bi-LSTM) to encode an input sequence. |
| max_pooling | bool, if it is True the max-pooling operation will be performed, else the last hidden state from the reccurent neural network will be taken.|
| type_of_weights | {"shared", "separate"}, use shared of separate weights to encode the context and response. |
| hidden_dim | int, a size of a hidden state if the "reccurent" parameter is set to the "lstm" or the half-size if the "reccurent" is set to the "bilstm".| 
| learning_rate | float, learning rate for training.    | 
| margin     | float, a margin to use in a triplet loss. | 
| load_path           | str, a path to a file from which model files will be loaded.    |
| save_path           | str, a path to a file where model files will be saved.    |
| interact_pred_num | int, first "interact_pred_num" best candidates for context and response to show in the "interact" regime. |
| **train**           | **parameters for training** |
| epochs              | int, a number of epochs for training. |
| batch_size          | int, a batch size for training. |
| metrics             | a list of metrics names , top-1 "r@1" recall, "r@2", "r@5" and the average position of the correct response among all response candidates "rank_response" are available for the model. | 
| validation_patience | int, for how many epochs the training can continue without improvement of the metric value on the validation set.           | 
| val_every_n_epochs  | int, a frequency of validation during training (validate every n epochs).       | 
