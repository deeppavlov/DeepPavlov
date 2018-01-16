[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Neural Model for Intent Classification

In this repo one can find code for training and infering intent classification model
that is presented as shallow-and-wide Convolutional Neural Network[1]. 
The model is multi-class and multi-label that means each text of a dataset 
can belong to several classes or not belong to any one.

Also there is presented pre-trained model for user intent classification for DSTC 2 dataset [CITE].
Intents for user replies are being constructed by IntentDataset (`deeppavlov/datasets/intent_dataset.py`).
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


## Usage

#### Config parameters:  

|   Parameter         |  Description                                                      | Set of Values                                 | Type    |
|---------------------|-------------------------------------------------------------------|-----------------------------------------------|---------|
| model_name          | method of the class that corresponds to the model                 | \[cnn_model, dcnn_model\] or any custom one   | str     |
| text_size           | length of each sample in words                                    |  (0,$\infty$)                                 | int     |
| confident_threshold | boundary value of belonging to a class                            |   \[0., 1.\]                                  | float   |
| kernel_sizes_cnn    | kernel sizes for shallow-and-wide and deep CNN model              |  i.e "3 3 3"                                  | str     |
| filters_cnn         | number(-s) of filters for shallow-and-wide (deep) CNN             | (0,\infty) or i.e "128 256"                   | int,str |  
| dense_size          | size of dense layer previous for classifying one                  |    (0,\infty)                                 | int     |
| lear_metrics        | learning metrics for training                                     | from keras.metrics                            | str     |
| lear_rate           | learning rate for training                                        |    (0,\infty)                                 | float   |
| lear_rate_decay     | learning rate decay for training                                  |    (0,\infty)                                 | float   |
| optimizer           | optimizer for training                                            | from keras.optimizers                         | str     |
| loss                | loss for training                                                 | from keras.losses                             | str     |
| coef_reg_cnn        | coefficient for kernel l2-regularizer for convolutional layers    |   \[0., 1.\]                                  | float   |
| coef_reg_den        | coefficient for kernel l2-regularizer for dense layers            |   \[0., 1.\]                                  | float   |
| dropout_rate        | dropout rate for training                                         |   \[0., 1.\]                                  | float   |
| epochs              | number of epochs for training                                     |   (0,\infty)                                  |  int    |
| batch_size          | batch size for training                                           |   (0,\infty)                                  |  int    |
| val_every_n_epochs  | frequency of validation during training (validate every n epochs) |  (0,\infty)                                   |  int    |
| verbose             | parameter whether to print training information or not            |  (True, False)                                | bool    |
| val_patience        | maximal number of validation loss increases before stop training  |   (0,\infty)                                  |  int    |
| classes_file        | file to save list of classes extracted from data                  | i.e "classes.txt"                             | str     |


      
#### Usage example


## Training


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