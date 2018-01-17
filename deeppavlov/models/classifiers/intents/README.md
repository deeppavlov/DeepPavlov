[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Neural Model for Intent Classification

In this repo one can find code for training and infering intent classification model
that is presented as shallow-and-wide Convolutional Neural Network[1]. 
The model is multi-class and multi-label that means each text of a dataset 
can belong to several classes or not belong to any one.

Also there is presented pre-trained model for user intent classification for DSTC 2 dataset [CITE].
DSTC 2 dataset does not initially contains information about intents, 
therefore, `IntentDataset` (`deeppavlov/datasets/intent_dataset.py`) instance artificially extracts 
intents for each user reply using information from acts and slots.
Below several examples of intent construction are given:

> System: "Hello, welcome to the Cambridge restaurant system. You can ask for restaurants by area, price range or food type. How may I help you?"
>
> User: "cheap restaurant"

In the original dataset this user reply has characteristics
`"goals": {"pricerange": "cheap"}, "db_result": null, "dialog-acts": [{"slots": [["pricerange", "cheap"]], "act": "inform"}]}`.
This message contains the only intent `inform_pricerange`.

> User: "thank you good bye",

In the original dataset this user reply has characteristics 
`"goals": {"food": "dontcare", "pricerange": "cheap", "area": "south"}, "db_result": null, "dialog-acts": [{"slots": [], "act": "thankyou"}, {"slots": [], "act": "bye"}]}`
This message contains two intents `(thankyou, bye)`.


## Usage of pre-trained model

To infer using console interface one can set parameter `MODEL_CONFIG_PATH='models/classifiers/intents/config.json'` 
in `deeppavlov/run_model.py`, and then run
```
python run_model.py 
```
Now user can enter a text string and get an intent (class a request belongs with):
```
:: hey! I want cheap chinese restaurant
>> ['inform_food' 'inform_pricerange']
```


#### Config parameters:  

|   Parameter         |  Description                                                      | Set of Values                                 | Type    |
|---------------------|-------------------------------------------------------------------|-----------------------------------------------|---------|
| model_name          | method of the class that corresponds to the model                 | \[cnn_model, dcnn_model\] or any custom one   | str     |
| text_size           | length of each sample in words                                    |  (0, ∞)                                       | int     |
| confident_threshold | boundary value of belonging to a class                            |   \[0., 1.\]                                  | float   |
| kernel_sizes_cnn    | kernel sizes for shallow-and-wide and deep CNN model              |  i.e "3 3 3"                                  | str     |
| filters_cnn         | number(-s) of filters for shallow-and-wide (deep) CNN             | (0,∞) or i.e "128 256"                        | int,str |  
| dense_size          | size of dense layer previous for classifying one                  |    (0,∞)                                      | int     |
| lear_metrics        | learning metrics for training                                     | from keras.metrics                            | str     |
| lear_rate           | learning rate for training                                        |    (0,∞)                                      | float   |
| lear_rate_decay     | learning rate decay for training                                  |    (0,∞)                                      | float   |
| optimizer           | optimizer for training                                            | from keras.optimizers                         | str     |
| loss                | loss for training                                                 | from keras.losses                             | str     |
| coef_reg_cnn        | coefficient for kernel l2-regularizer for convolutional layers    |   \[0., 1.\]                                  | float   |
| coef_reg_den        | coefficient for kernel l2-regularizer for dense layers            |   \[0., 1.\]                                  | float   |
| dropout_rate        | dropout rate for training                                         |   \[0., 1.\]                                  | float   |
| epochs              | number of epochs for training                                     |   (0,∞)                                       |  int    |
| batch_size          | batch size for training                                           |   (0,∞)                                       |  int    |
| val_every_n_epochs  | frequency of validation during training (validate every n epochs) |  (0,∞)                                        |  int    |
| verbose             | parameter whether to print training information or not            |  (True, False)                                | bool    |
| val_patience        | maximal number of validation loss increases before stop training  |   (0,∞)                                       |  int    |
| classes_file        | file to save list of classes extracted from data                  | i.e "classes.txt"                             | str     |

## Training model

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