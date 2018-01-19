[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Neural Model for Intent Classification

In this repository one can find code for training and infering from intent classification model
that is presented as shallow-and-wide Convolutional Neural Network[1]. 
The model is multi-class and multi-label that means each text of a dataset 
can belong to several classes.

Also there is presented pre-trained model for user intent classification for DSTC 2 dataset [CITE].
DSTC 2 dataset does not initially contains information about intents, 
therefore, `IntentDataset` (`deeppavlov/datasets/intent_dataset.py`) instance artificially extracts 
intents for each user reply using information from acts and slots.
Below several examples of intent construction are given:

> System: "Hello, welcome to the Cambridge restaurant system. You can ask for restaurants by area, price range or food type. How may I help you?"
>
> User: "cheap restaurant"

In the original dataset this user reply has characteristics
```
"goals": {"pricerange": "cheap"}, 
"db_result": null, 
"dialog-acts": [{"slots": [["pricerange", "cheap"]], "act": "inform"}]}
```
This message contains the only intent `inform_pricerange`.

> User: "thank you good bye",

In the original dataset this user reply has characteristics 
```
"goals": {"food": "dontcare", "pricerange": "cheap", "area": "south"}, 
"db_result": null, 
"dialog-acts": [{"slots": [], "act": "thankyou"}, {"slots": [], "act": "bye"}]}
```
This message contains two intents `(thankyou, bye)`.


## Infer from pre-trained model

To infer using console interface one can set parameter `MODEL_CONFIG_PATH='models/classifiers/intents/config_infer.json'` 
in `deeppavlov/run_model.py` (minimal required set of parameters to infer pre-trained model),
and then run
```
python run_model.py 
```
Now user can enter a text string and get intents (classes which a request belongs with):
```
:: hey! I want cheap chinese restaurant
>> ['inform_food' 'inform_pricerange']
```

## Train model

One of the main constituents of model is a configuration file. Below the table with description of parameters is presented.

#### Configuration parameters:  

| Structure      |   Parameter         |  Description                                                      | Set of Values                                 | Type    |
|----------------|---------------------|-------------------------------------------------------------------|-----------------------------------------------|---------|
| dataset_reader |||||
|   name              | registered name of dataset reader                                 | dstc2_datasetreader, classification_datasetreader |
|   data_path         | directory where data files are located                            | directory                                     | 
|   data_types        | which data types is presented in data_path (only for `classification_datasetreader`) | list of fields, i.e ["train", "valid", "test"]| 
| dataset|||||
|   name              | registered name of dataset                                        | "intent_dataset", classification_dataset"     | 
|   seed              | seed for batch generator                                          | int from \[0., 1.\]                           |
|   fields_to_merge   | list of fields to merge                                           | list of fields, i.e ["train", "valid", "test"]| 
|   merged_field      | name of field to which save merged fields                         | field, i.e "train", "valid", "test"           | 
|   field_to_split    | name of field to split                                            | field, i.e "train", "valid", "test"           | 
|   split_fields      | list of fields to which save splitted field                       | list of fields, i.e ["train", "valid", "test"]|
|   split_proportions | list of corresponding proportions for splitting                   | list of floats each of which is in  \[0., 1.\]|
| vocabs.classes_vocab |||||
| name                | registered name of vocab                                          | "default_vocab"                               | 
|  inputs             | whether to create vocab over x and/or y fields of dataset         | list of "x" and/or "y"                        |
|  level              | whether to considered char or token level                         | "char", "token"                               |
| model_path          | path to file where vocab with classes will be saved               | filename                                      |
| train_now           | whether to train vocab or not                                     | true \| false                                 |
| model.embedder |||||
| name                | registered name of embedder                                       | "fasttext"                                    |
| model_path          | path to file where binary embedding model is located              | filename                                      | 
| emb_module          | fasttext library to use                                           | "fasttext", "pyfasttext", "gensim"            | 
| dim                 | dimension of embeddings                                           |  int from (0,∞)                               | 
| model.tokenizer|||||
| name                | registered name of tokenizer                                      | "nltk_tokenizer"                              | 
| tokenizer           | tokenizer from nltk.tokenize to use                               | any method from nltk.tokenize                 |  
| model         |||||
| model_name          | method of the class KerasIntentModel that corresponds to the model| \[cnn_model, dcnn_model\] or any custom one   | 
| text_size           | length of each sample in words                                    |  int from (0, ∞)                              | 
| confident_threshold | boundary value of belonging to a class                            |   float from \[0., 1.\]                       | 
| kernel_sizes_cnn    | kernel sizes for shallow-and-wide and deep CNN model              |  list of int from (0,∞)                       | 
| filters_cnn         | number(-s) of filters for shallow-and-wide (deep) CNN             | list of int from (0,∞)                        | 
| dense_size          | size of dense layer previous for classifying one                  |    integer from (0,∞)                         | 
| lear_metrics        | learning metrics for training                                     | from keras.metrics                            | 
| lear_rate           | learning rate for training                                        |    float from (0,∞)                           | 
| lear_rate_decay     | learning rate decay for training                                  |    float from (0,∞)                           | 
| optimizer           | optimizer for training                                            | from keras.optimizers                         |
| loss                | loss for training                                                 | from keras.losses                             |
| coef_reg_cnn        | coefficient for kernel l2-regularizer for convolutional layers    |   float from \[0., 1.\]                       |
| coef_reg_den        | coefficient for kernel l2-regularizer for dense layers            |   float from \[0., 1.\]                       |
| dropout_rate        | dropout rate for training                                         |   float from \[0., 1.\]                       |
| epochs              | number of epochs for training                                     |   int form from (0,∞)                         |
| batch_size          | batch size for training                                           |   int from (0,∞)                              |
| val_every_n_epochs  | frequency of validation during training (validate every n epochs) |  int from (0,∞)                               | 
| verbose             | parameter whether to print training information or not            |  true \| false                                | 
| val_patience        | maximal number of validation loss increases before stop training  |  int from (0,∞)                               | 
| classes_file        | file to save list of classes extracted from data                  | i.e "classes.txt"                             | 

### Training on DSTC 2

To train model again or with other parameters 
 the only actions are to set parameter `train_now` to `True` in `config.json` and
 set `model_path` to the directory where trained model will be saved (it will be created if does not exist). 
 All other parameters of model as well as fasttext model could be changed. 
 Then training could be run in the following way:
```
python run_model.py 
```

### Training on other dataset

Constructing intents from DSTC 2 makes `IntentDataset` a bit difficult, therefore, 
another dataset reader `ClassificationDatasetReader` and dataset `ClassificationDataset` 
 are also provided in `deeppavlov/dataset_readers` and `deeppavlov/datasets`.
 
Training data files `train.csv`, `valid.csv` should be presented in the following form:

| text         |class_0|class_1|class_2|class_3| ...|
|------------- |:-----:|:-----:|:-----:|:-----:|:--:|
| text_0       | 1     | 0     | 0     |0      |... |
| text_1       | 0     | 0     | 1     |0      |... |
| text_2       | 0     | 1     | 0     |0      |... |
| text_3       | 1     | 0     | 1     |0      |... |
| ...          | ...   | ...   | ...   |...    |... ||


To train model the only actions are to set parameter `train_now` to `True` in `config.json`,
 set `data_path` to the directory containing `train.csv`, `valid.csv`, and if necessary, `test.csv`,
 set `model_path` to the directory where trained model will be saved. 
 All other parameters of model as well as fasttext model could be changed. 
 Then training could be run in the same way:
 ```
python run_model.py 
```

## Comparison



|             Model                          |  AUC-ROC  | F-measure | 
|--------------------------------------------|-----------|-----------|
| 1                                          | 2         |           | 
|                                            |           |           | 
|                                            |           |           | 
|                                            |           |           |

## Ways to improve

* 
*
*

# References

[1] Kim Y. Convolutional neural networks for sentence classification //arXiv preprint arXiv:1408.5882. – 2014.

[2] P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information.