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

To infer using console interface one have to run
```
python deep.py interact configs/intents/intents_dstc2.json
```
or
```
python deep.py interactbot configs/intents/intents_dstc2.json -t <TELEGRAM_TOKEN>
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
| **dataset_reader**  | instance to read datasets from files |
|   name              | registered name of dataset reader   <br />*SetOfValues*: "dstc2_datasetreader", "classification_datasetreader" |
|   data_path         | directory where data files are located                          | 
|   data_types        | which data types is presented in data_path (only for `classification_datasetreader`) *SetOfValues*: list of fields, i.e ["train", "valid", "test"]| 
| **dataset**         | instance to provide models with data in the standard form (each example is a tuple (x, y) where x and y could be numbers, binaries, lists or strings) |
|   name              | registered name of dataset        <br />*SetOfValues*:  "intent_dataset", classification_dataset"     | 
|   seed              | seed for batch generator              |
|   fields_to_merge   | list of fields to merge                <br />*SetOfValues*: list of fields, i.e ["train", "valid", "test"]| 
|   merged_field      | name of field to which save merged fields      <br />*SetOfValues*:  field, i.e "train", "valid", "test"           | 
|   field_to_split    | name of field to split                         <br />*SetOfValues*:  field, i.e "train", "valid", "test"           | 
|   split_fields      | list of fields to which save splitted field     <br />*SetOfValues*:  list of fields, i.e ["train", "valid", "test"]|
|   split_proportions | list of corresponding proportions for splitting  <br />*SetOfValues*:  list of floats each of which is in  \[0., 1.\]|
| **chainer**         | chainer is a structure that receives tuple `(in, in_y)` and produces `out`     |
|in                   | user-defined name of input (or list of names in case of different inputs) <br />*SetOfValues*: list of names, i.e ["x"], ["x0", "x1"] |
|in_y                 | user-defined name of input targets (or list of names in case of different input targets) <br />*SetOfValues*: list of names, i.e ["y"], ["y0", "y1"] |
|out                  | user-defined name of output (or list of names in case of different outputs) <br />*SetOfValues*: list of names, i.e ["y_pred"], ["y_pred0", "y_pred1"] |
| **chainer.pipe**    | list that contains sequence of models (including vocabs, preprocessors, postprocessors etc.) |
| **chainer.pipe.vocabs** | instances to create vocabularies over x and y |
| id                  | key name for considered model for further references |
| name                | registered name of vocab    <br />*SetOfValues*: "default_vocab"  | 
|  fit_on             | whether to create vocab over x and/or y fields of dataset  <br />*SetOfValues*: list of "x" and/or "y"    |
|  level              | whether to considered char or token level     <br />*SetOfValues*: "char", "token"   |
| load_path           | path to file which vocab with classes will be loaded from    |
| save_path           | path to file where vocab with classes will be saved    |
| **chainer.pipe.model** |  |
| in                  | inputs for the model <br />*SetOfValues*: list of names from chainer.in, chainer.in_y or outputs of previous models |
| in_y                | input targets for the model, obligatory for training <br />*SetOfValues*: list of names from chainer.in, chainer.in_y or outputs of previous models  |
| out                 | outputs for the model <br />*SetOfValues*: list of names |
| main                | determines main part of the pipe |
| name                | registered name of model  | 
| load_path           | path to file which model files will be loaded from    |
| save_path           | path to file where model files will be saved    |
| classes             | list of classes names. In this case they could be simply obtained from `classes_vocab.keys()` method |
| opt                 | dictionary of parameters for model |
| opt.model_name      | method of the class KerasIntentModel that corresponds to the model <br />*SetOfValues*: "cnn_model", "dcnn_model"   | 
| opt.text_size       | length of each sample in words      | 
| opt.confident_threshold | boundary value of belonging to a class  <br />*SetOfValues*: \[0., 1.\]                       | 
| opt.kernel_sizes_cnn| kernel sizes for shallow-and-wide and deep CNN model        | 
| opt.filters_cnn     | number(-s) of filters for shallow-and-wide (deep) CNN   | 
| opt.dense_size      | size of dense layer previous for classifying one    | 
| opt.lear_rate       | learning rate for training    | 
| opt.lear_rate_decay | learning rate decay for training          | 
| opt.optimizer       | optimizer for training    <br />*SetOfValues*: any method from keras.optimizers                         |
| opt.loss            | loss for training       <br />*SetOfValues*: any method from keras.losses                             |
| opt.coef_reg_cnn    | coefficient for kernel l2-regularizer for convolutional layers   |
| opt.coef_reg_den    | coefficient for kernel l2-regularizer for dense layers  |
| opt.dropout_rate    | dropout rate for training    |
| **model.embedder**            | instance to produce word vectors |
| embedder.name       | registered name of embedder  <br />*SetOfValues*:"fasttext"   |
| embedder.load_path  | path to file which embedding binary file will be loaded from    |
| embedder.emb_module | fasttext library to use  <br />*SetOfValues*: "fasttext", "pyfasttext", "gensim"            | 
| embedder.dim        | dimension of embeddings    | 
| **model.tokenizer**           | instance to tokenize texts |
| tokenizer.name      | registered name of tokenizer <br />*SetOfValues*: "nltk_tokenizer"                              | 
| tokenizer.tokenizer | tokenizer from nltk.tokenize to use  <br />*SetOfValues*:  any method from nltk.tokenize    |  
| **train**            | |
| epochs              | number of epochs for training    |
| batch_size          | batch size for training    |
| metrics             | learning metrics for training  <br />*SetOfValues*: any method from from keras.metrics         | 
| validation_patience | maximal number of validation loss increases before stop training           | 
| val_every_n_epochs  | frequency of validation during training (validate every n epochs)       | 
| val_every_n_batches | frequency of validation during training (validate every n batches)      | 
| show_examples       | parameter whether to print training information or not        | 

### Train on DSTC-2

To train model again or with other parameters on DSTC-2 dataset,
 set `save_path` to the directory where trained model will be saved 
 (pre-trained model will be loaded if `load_path` is provided and files exist, and it will be created from scratch otherwise).
 All other parameters of model as well as embedder and tokenizer could be changed. 
 Then training could be run in the following way:
```
python deep.py train configs/intents/intents_dstc2.json
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
* set `data_path` to the directory where `train.csv` will be downloaded to,
* set `save_path` to the directory where trained model will be saved to, 
* set all other parameters of model as well as embedder and tokenizer to desired ones.

Then training could be run in the same way:
```
python deep.py train configs/intents/intents_snips.json
```

**Current `intents_snips.json` implies intent recognition for SNIPS benchmark dataset [2] 
that was restored in `.csv` format and will be downloaded automatically.**

**Important: there are not provided special embedding binary file and pre-trained model files for SNIPS dataset.
Please, provide you own embedding binary file to train model because embedding file trained on DSTC-2 dataset is not the best choice for this task.**

## Comparison

As no one had published intent recognition for DSTC-2 data, 
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

* One can train another embeddings using FastText [4] that are more appropriate for considered datasets.
* All the parameters have to be tuned for training.

# References

[1] Kim Y. Convolutional neural networks for sentence classification //arXiv preprint arXiv:1408.5882. – 2014.

[2] https://github.com/snipsco/nlu-benchmark

[3] https://www.slideshare.net/KonstantinSavenkov/nlu-intent-detection-benchmark-by-intento-august-2017

[4] P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information.
