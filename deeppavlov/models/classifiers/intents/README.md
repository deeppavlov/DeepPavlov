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


## Usage of pre-trained model

To infer using console interface one can set parameter `MODEL_CONFIG_PATH='models/classifiers/intents/config.json'` 
in `deeppavlov/run_model.py`, and then run
```
python run_model.py 
```
Now user can enter a text string and get a vector of probabilities to belong to each class (to contain an intent):
```
:: hey! I want cheap chinese restaurant
>> [  5.27572702e-04   7.23480759e-03   1.76249957e-03   1.09066057e-03
   3.06853256e-03   3.45652481e-03   1.03214942e-03   5.49707853e-04
   6.54116739e-03   2.15114257e-03   9.90729392e-01   6.34791562e-04
   9.66436625e-01   2.64307763e-03   6.54205796e-04   8.30116624e-04
   1.85919739e-03   2.43586794e-04   1.80137530e-03   1.55517610e-03
   8.78513150e-04   9.00576240e-04   7.22794677e-04   1.37247320e-03
   3.60768143e-04   5.41293994e-04   8.34978826e-04   3.10848374e-03
   6.91671812e-06]
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

If one wants to train a model on some other data, please

To train model the only action is to set parameter `train_now` to `True` in `config.json`,
 and set `model_path` to the directory where trained model will be saved. All other parameters of model 
 as well as fasttext model could be changed. Then training could be run in the same way:
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