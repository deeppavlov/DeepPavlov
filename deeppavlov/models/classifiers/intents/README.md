[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Neural Model for Intent Classification

In this repository one can find code for training and using the intent classification model
which is implemented as a shallow-and-wide Convolutional Neural Network[1]. 
The model is multi-class and multi-label, which means that each text in a dataset 
can belong to several classes.

We also have a pre-trained model for user intent classification for DSTC 2 dataset.
DSTC 2 dataset does not initially contain information about intents, 
therefore, `IntentDataset` (`deeppavlov/datasets/intent_dataset.py`) instance extracts 
artificial intents for each user reply using information from acts and slots.

Below we give several examples of intent construction:

> System: "Hello, welcome to the Cambridge restaurant system. You can ask for restaurants by area, price range or food type. How may I help you?"
>
> User: "cheap restaurant"

In the original dataset this user reply has characteristics
```
"goals": {"pricerange": "cheap"}, 
"db_result": null, 
"dialog-acts": [{"slots": [["pricerange", "cheap"]], "act": "inform"}]}
```
This message contains only one intent: `inform_pricerange`.

> User: "thank you good bye",

In the original dataset this user reply has characteristics 
```
"goals": {"food": "dontcare", "pricerange": "cheap", "area": "south"}, 
"db_result": null, 
"dialog-acts": [{"slots": [], "act": "thankyou"}, {"slots": [], "act": "bye"}]}
```
This message contains two intents `(thankyou, bye)`.


## Infer from pre-trained model

To use a pre-trained model for inference you should run the following command:
```
python deep.py interact configs/intents/intents_dstc2.json
```
or
```
python deep.py interactbot configs/intents/intents_dstc2.json -t <TELEGRAM_TOKEN>
```
For 'interactbot' mode you should specify a Telegram bot token in `-t` parameter or in the `TELEGRAM_TOKEN` 
environment variable.


Now user can enter a text string and get intents (classes which a request belongs to):
```
:: hey! I want cheap chinese restaurant
>> ['inform_food' 'inform_pricerange']
```

## Train model

You can find examples of config files [here](../../../configs/intents).

All parameters for [intents_dstc2.json](../../../configs/intents/intents_dstc2.json) config file are presented in the table below.

#### Configuration parameters:  

|   Parameter         |  Description                                                      | 
|---------------------|-------------------------------------------------------------------|
| **dataset_reader**  | **an object that reads datasets from files** |
|   name              | registered name of the dataset reader   <br />*SetOfValues*: "dstc2_datasetreader", "classification_datasetreader" |
|   data_path         | directory where data files are stored                          | 
| **dataset**         | **an object that provides models with data in the standard form (each example is a tuple (x, y) where x and y could be numbers, booleans, lists or strings)** |
|   name              | registered name of the dataset        <br />*SetOfValues*:  "intent_dataset", classification_dataset"     | 
|   seed              | seed for the batch generator              |
|   fields_to_merge   | list of fields to merge                <br />*SetOfValues*: list of fields, i.e ["train", "valid", "test"]| 
|   merged_field      | name of the field where the merged fields should be saved      <br />*SetOfValues*:  field, i.e "train", "valid", "test"           | 
|   field_to_split    | name of the field to split                         <br />*SetOfValues*:  field, i.e "train", "valid", "test"           | 
|   split_fields      | list of fields where the splitted field should be saved     <br />*SetOfValues*:  list of fields, i.e ["train", "valid", "test"]|
|   split_proportions | list of corresponding proportions for splitting  <br />*SetOfValues*:  list of floats each of which is in  \[0., 1.\]|
| **chainer**         | **chainer is a structure that receives tuples `(in, in_y)` and produces `out`**     |
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

### Train on DSTC-2

To re-train a model or train it with different parameters on DSTC-2 dataset,
 you should set `save_path` to a directory where the trained model will be saved 
 (pre-trained model will be loaded if `load_path` is provided and files exist, otherwise it will be created from scratch).
 All other parameters of the model as well as embedder and tokenizer could be changed. 
 Then training can be run in the following way:
```
python deep.py train configs/intents/intents_dstc2.json
```

### Train on other datasets

Constructing intents from DSTC 2 makes `IntentDataset` difficult to use.
Therefore, we also provide another dataset reader `ClassificationDatasetReader` and dataset `ClassificationDataset`
to work with `.csv` files. These classes are described in `deeppavlov/dataset_readers` and `deeppavlov/datasets`.
 
Training data file `train.csv` (and `valid.csv`, if exists) should be in the following format:

| text         |intents|
|------------- |:-----:|
| text_0       | intent_0     |
| text_1       | intent_0     |
| text_2       | intent_1,intent_2     |
| text_3       | intent_1,intent_0    | 
| ...          | ...   ||


To train model you should
* set `data_path` to the directory to which `train.csv` should be downloaded,
* set `save_path` to the directory where the trained model should be saved, 
* set all other parameters of model as well as embedder and tokenizer to desired ones.

Then the training can be run in the same way:
```
python deep.py train configs/intents/intents_snips.json
```

**The current version of `intents_snips.json` contains parameters for intent recognition for SNIPS benchmark dataset [2] 
that was restored in `.csv` format and will be downloaded automatically.**

**Important: we do not provide any special embedding binary file and pre-trained model files for SNIPS dataset.
In order to train the model you should provide you own embedding binary file, because embedding file trained on DSTC-2 dataset is not the best choice for this task.**

## Comparison

As no one had published intent recognition for DSTC-2 data, 
the comparison of the presented model is given on **SNIPS** dataset. 
The evaluation of model scores was conducted in the same way as in [3] to compare with the results from the report of the authors of the dataset.
The results were achieved with tuning of parameters.

|             Model    | AddToPlaylist | BookRestaurant | GetWheather | PlayMusic | RateBook | SearchCreativeWork | SearchScreeningEvent | 
|----------------------|---------------|----------------|-------------|-----------|----------|--------------------|----------------------|
| api.ai               |  0.9931       |  0.9949        |  0.9935     |   0.9811  | 0.9992   | 0.9659             | 0.9801               |
| ibm.watson           |  0.9931       |  0.9950        | 0.9950      | 0.9822    | 0.9996   | 0.9643             |  0.9750              |
| microsoft.luis       |  0.9943       |   0.9935       |   0.9925    |  0.9815   | 0.9988   | 0.9620             |  0.9749              |
| wit.ai               |    0.9877     |   0.9913       |   0.9921    |  0.9766   | 0.9977   | 0.9458             |  0.9673              |
| snips.ai             |   0.9873      |   0.9921       |   0.9939    |  0.9729   | 0.9985   | 0.9455             |  0.9613              |
| recast.ai            |   0.9894      |   0.9943       |   0.9910    |  0.9660   | 0.9981   | 0.9424             |  0.9539              |
| amazon.lex           |   0.9930      |  0.9862        |   0.9825    |  0.9709   | 0.9981   | 0.9427             |  0.9581              |
|||||||||
| Shallow-and-wide CNN |  **0.9956**   |   **0.9973**   |   **0.9968**| **0.9871**| **0.9998**| **0.9752**     |   **0.9854**           |


## How to improve the performance

* You can use FastText [4] to train embeddings that are better suited for considered datasets.
* All the parameters should be tuned on the validation set.

# References

[1] Kim Y. Convolutional neural networks for sentence classification //arXiv preprint arXiv:1408.5882. – 2014.

[2] https://github.com/snipsco/nlu-benchmark

[3] https://www.slideshare.net/KonstantinSavenkov/nlu-intent-detection-benchmark-by-intento-august-2017

[4] P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information.
