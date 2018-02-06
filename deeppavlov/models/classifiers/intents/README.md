[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Neural Model for Intent Classification

In this repository one can find code for training and infering from intent classification model
that is presented as shallow-and-wide Convolutional Neural Network[1]. 
The model is multi-class and multi-label that means each text of a dataset 
can belong to several classes.

Also there is presented pre-trained model for user intent classification for DSTC 2 dataset.
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

To infer using console interface one have tp set parameter `train_now` to `false` 
in `configs/intents/config_dstc2.json` and then run
```
python deep.py interact configs/intents/config_dstc2.json
```
or
```
python deep.py interactbot configs/intents/config_dstc2_infer.json
```
For 'interactbot' mode you should specify Telegram bot token in `-t` parameter or in `TELEGRAM_TOKEN` 
environment variable.


Now user can enter a text string and get intents (classes which a request belongs with):
```
:: hey! I want cheap chinese restaurant
>> ['inform_food' 'inform_pricerange']
```

## Train model

One of the main constituents of model is a configuration file. 
Below the table with description of parameters is presented.

#### Configuration parameters:  

|   Parameter         |  Description                                                      | 
|---------------------|-------------------------------------------------------------------|
| **dataset_reader** ||
|   name              | registered name of dataset reader   <br />*SetOfValues*: "dstc2_datasetreader", "classification_datasetreader" |
|   data_path         | directory where data files are located                          | 
|   data_types        | which data types is presented in data_path (only for `classification_datasetreader`) *SetOfValues*: list of fields, i.e ["train", "valid", "test"]| 
| **dataset** ||
|   name              | registered name of dataset        <br />*SetOfValues*:  "intent_dataset", classification_dataset"     | 
|   seed              | seed for batch generator              |
|   fields_to_merge   | list of fields to merge                <br />*SetOfValues*: list of fields, i.e ["train", "valid", "test"]| 
|   merged_field      | name of field to which save merged fields      <br />*SetOfValues*:  field, i.e "train", "valid", "test"           | 
|   field_to_split    | name of field to split                         <br />*SetOfValues*:  field, i.e "train", "valid", "test"           | 
|   split_fields      | list of fields to which save splitted field     <br />*SetOfValues*:  list of fields, i.e ["train", "valid", "test"]|
|   split_proportions | list of corresponding proportions for splitting  <br />*SetOfValues*:  list of floats each of which is in  \[0., 1.\]|
| **vocabs.classes_vocab** ||
| name                | registered name of vocab    <br />*SetOfValues*: "default_vocab"  | 
|  inputs             | whether to create vocab over x and/or y fields of dataset  <br />*SetOfValues*: list of "x" and/or "y"    |
|  level              | whether to considered char or token level     <br />*SetOfValues*: "char", "token"   |
| load_path           | path to file which vocab with classes will be loaded from    |
| save_path           | path to file where vocab with classes will be saved    |
| train_now           | whether to train vocab or not  |
| **model**           |
| name                | registered name of model  | 
| model_name          | method of the class KerasIntentModel that corresponds to the model <br />*SetOfValues*: "cnn_model", "dcnn_model"   | 
| text_size           | length of each sample in words      | 
| confident_threshold | boundary value of belonging to a class  <br />*SetOfValues*: \[0., 1.\]                       | 
| kernel_sizes_cnn    | kernel sizes for shallow-and-wide and deep CNN model        | 
| filters_cnn         | number(-s) of filters for shallow-and-wide (deep) CNN   | 
| dense_size          | size of dense layer previous for classifying one    | 
| lear_metrics        | learning metrics for training  <br />*SetOfValues*: any method from from keras.metrics                            | 
| lear_rate           | learning rate for training    | 
| lear_rate_decay     | learning rate decay for training          | 
| optimizer           | optimizer for training    <br />*SetOfValues*: any method from keras.optimizers                         |
| loss                | loss for training       <br />*SetOfValues*: any method from keras.losses                             |
| coef_reg_cnn        | coefficient for kernel l2-regularizer for convolutional layers   |
| coef_reg_den        | coefficient for kernel l2-regularizer for dense layers  |
| dropout_rate        | dropout rate for training    |
| epochs              | number of epochs for training    |
| batch_size          | batch size for training    |
| val_every_n_epochs  | frequency of validation during training (validate every n epochs)       | 
| verbose             | parameter whether to print training information or not        | 
| val_patience        | maximal number of validation loss increases before stop training           | 
| load_path           | path to file which model files will be loaded from    |
| save_path           | path to file where model files will be saved    |
| **model.embedder** ||
| name                | registered name of embedder  <br />*SetOfValues*:"fasttext"   |
| load_path           | path to file which embedding binary file will be loaded from    |
| emb_module          | fasttext library to use  <br />*SetOfValues*: "fasttext", "pyfasttext", "gensim"            | 
| dim                 | dimension of embeddings    | 
| **model.tokenizer** ||
| name                | registered name of tokenizer <br />*SetOfValues*: "nltk_tokenizer"                              | 
| tokenizer           | tokenizer from nltk.tokenize to use  <br />*SetOfValues*:  any method from nltk.tokenize                 |  

### Train on DSTC 2

To train model again or with other parameters on DSTC 2 data
 the only actions are to set parameter `train_now` to `true` in `configs/intents/config_dstc2.json`,
 set `model_path` to the directory where trained model will be saved 
 (it will be loaded if model exists, and it will be created otherwise).
 All other parameters of model as well as embedder and tokenizer could be changed. 
 Then training could be run in the following way:
```
python deep.py train configs/intents/config_dstc2.json
```

### Train on other data

Constructing intents from DSTC 2 makes `IntentDataset` a bit difficult.
Therefore, another dataset reader `ClassificationDatasetReader` and dataset `ClassificationDataset` 
to work with `.csv` files are also provided in `deeppavlov/dataset_readers` and `deeppavlov/datasets`.
 
Training data files `train.csv` (and, if exists, `valid.csv`) should be presented in the following form:

| text         |intents|
|------------- |:-----:|
| text_0       | intent_0     |
| text_1       | intent_0     |
| text_2       | intent_1,intent_2     |
| text_3       | intent_1,intent_0    | 
| ...          | ...   ||


To train model one should 
* set parameter `train_now` to `true` in `configs/intents/config_snips.json`,
* set `data_path` to the directory where `train.csv` will be downloaded to,
* set `model_path` to the directory where trained model will be saved to, 
* set all other parameters of model as well as embedder and tokenizer to desired ones.

Then training could be run in the same way:
```
python deep.py train configs/intents/config_snips.json
```

**Current `config_snips.json` implies intent recognition for SNIPS benchmark dataset [2] 
that was restored in `.csv` format and will be downloaded automatically.**

**Important: there are not provided embedding binary file and pre-trained model files for SNIPS dataset.
Please, provide you own embedding binary file to train model.**

## Comparison

As no one had published intent recognition for DSTC 2 data, 
comparison of the presented model is given on **SNIPS** dataset. 
Estimation of model scores was conducted in the same way as in [3] to compare with results from the presentation.
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


## Ways to improve

* One can train the other embeddings using FastText [4] that are more appropriate for the considered dataset.
* All the parameters have to be tuned for training.

# References

[1] Kim Y. Convolutional neural networks for sentence classification //arXiv preprint arXiv:1408.5882. – 2014.

[2] https://github.com/snipsco/nlu-benchmark

[3] https://www.slideshare.net/KonstantinSavenkov/nlu-intent-detection-benchmark-by-intento-august-2017

[4] P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information.
