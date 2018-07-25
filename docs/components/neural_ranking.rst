Neural Model for Ranking
========================

There is a task in NLP to retrieve the context closest semantically or
the most
suitable response to a given context from some context (response)
database.
The code in this repository uses a deep learning
approach to address the question answer selection task. Currently, a
basic
model is implemented with bidirectional long short-term memory
(biLSTM), with max pooling and without attention. The model was
trained on the dataset:
`InsuranceQA V1 <https://github.com/shuzi/insuranceQA>`__:

The distinguishing feature of the model is the use of triplet loss [1,
2].
This loss has a margin hyperparameter, which usually ranges from 0.01
to 0.2.
It is required to provide positive and negative response candidates
for each context
from the dataset to train the model.
Sampling of negative candidates can be performed globally from the
whole response set
or from pools of responses predefined separately for each context.
The same is true for validation and test, i.e. validation and test
can be carried out for the entire set of answers or
for the answers pools selected separately for each context.
There is a possibility in the model to encode contexts and responses
with biLSTM layers
having shared or separate weights.

Each train data sample for feeding the model is arranged as follows:

::

    {'context': [21507, 4696, 5843, 13035, 1628, 20923, 16083, 18690], 'response': 7009, 'pos_pool': [7009, 7010], 'neg_pool': None}

The context has a "context" key in the data sample.
It is represented by a list of integers which are keys
that give the list of tokens using the dictionary "integer–token".
The correct response has the "response" key in the sample,
its value is always a single integer.
The list of possible correct responses (there may be several) can be
obtained
with the "pos\_pool" key.
The value of the "response" should be equal to the one item from the
list
obtained using the "pos\_pool" key.
The list of possible negative responses (there can be a lot of them,
100–10000) is represented by the key "neg\_pool".
Its value is None, when global sampling is used, or the list of fixed
length, when sampling from predefined negative responses is used.
It is important that values in "pos\_pool" and "negative\_pool" do
not overlap.
Single responses in "response", "pos\_pool", "neg\_pool" are
represented
by single integers that give lists of integers
using the dictionary "integer–list of integers".
These lists of integers can be converted to lists of tokens with
the same dictionary "integer–token" which is used for contexts.
The additional "integer–list of integers" vocabulary is used
to not store all possible negative responses in the form of
sequences.
Validation and test data samples representation are almost the same
as the train samples one shown above.

Infer from pre-trained model
----------------------------

To use the model pre-trained on the InsuranceQA V1 dataset for
inference, one should run
the following command:

::

    python -m deeppavlov interact deeppavlov/configs/ranking/ranking_insurance.json

Now user can enter a text of context and get relevant contexts and
responses:

::

    :: how much to pay for auto insurance?
    >> {'contexts': ['how much can I expect pay for auto insurance', 'how much will insurance pay for my total car', 'how much can I expect pay in car insurance'], 'responses': ['the cost of auto insurance be based on several factor include your driving record , claim history , type of vehicle , credit score where you live and how far you travel to and from work I will recommend work with an independent agent who can shop several company find the good policy for you', 'there be not any absolute answer to this question rate for auto insurance coverage can vary greatly from carrier to carrier and from area to area contact local agent in your area find out about coverage availablity and pricing within your area look for an agent that you be comfortable working with as they will be the first last point of contact in most instance', 'the cost of auto insurance coverage for any vehicle or driver can vary greatly thing that effect your auto insurance rate be geographical location , vehicle , age (s) of driver (s) , type of coverage desire , motor vehicle record of all driver , credit rating of all driver and more contact a local agent get a quote a quote cost nothing but will let you know where your rate will']}

Train model
-----------

To train the model on the InsuranceQA dataset one should run the
command:

::

    python -m deeppavlov interact deeppavlov/configs/ranking/ranking_insurance.json

All parameters that can be set for the model (for example, see
``deeppavlov/configs/ranking/ranking_insurance.json>``)
are:

Configuration parameters:
^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameter                      | Description                                                                                                                                                                                                                                                    |
+================================+================================================================================================================================================================================================================================================================+
| **dataset\_reader**            | **reads datasets from files.**                                                                                                                                                                                                                                 |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| name                           | str, a registered name of the dataset reader.                                                                                                                                                                                                                  |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| data\_path                     | str, a directory where data files are stored.                                                                                                                                                                                                                  |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **dataset\_iterator**          | **provides models with data.**                                                                                                                                                                                                                                 |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| name                           | str, a registered name of the dataset.                                                                                                                                                                                                                         |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| seed                           | int or None (default=None), a seed for a batch generator.                                                                                                                                                                                                      |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| len\_vocab                     | int, a size of "integer–list of integers" vocab with all possible responses                                                                                                                                                                                    |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| sample\_candiates              | {"global", "pool"}, a method of negative sampling in train data. If "pool", negative candidates for each data sample should be provided. If "global", negative sampling over the whole data will be performed.                                                 |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| sample\_candiates\_valid       | {"global", "pool"}, a method of selecting\_candidates for ranking in valid data. If "pool", candidates for ranking for each data sample should be provided. If "global", all data samples will be taken as candidates for ranking.                             |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| sample\_candiates\_test        | {"global", "pool"}, a method of selecting\_candidates for ranking in valid data. If "pool", candidates for ranking for each data sample should be provided. If "global", all data samples will be taken as candidates for ranking.                             |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| num\_negative\_samples         | int, a number of negative samples to use if "sample\_candiates" is set to "pool".                                                                                                                                                                              |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| num\_ranking\_samples\_valid   | int, a number of negative samples to use if "sample\_candiates\_valid" is set to "pool".                                                                                                                                                                       |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| num\_ranking\_samples\_test    | int, a number of negative samples to use if "sample\_candiates\_test" is set to "pool".                                                                                                                                                                        |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **chainer**                    | **pipeline from heterogeneous components.**                                                                                                                                                                                                                    |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| in                             | list of str, a user-defined list of input names, i.e ["x"], ["x0", "x1"].                                                                                                                                                                                      |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| in\_y                          | list of str, a user-defined list of input target names , i.e ["y"], ["y0", "y1"].                                                                                                                                                                              |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| out                            | list of str, a user-defined list of output names, i.e. ["y\_pred"], ["y\_pred0", "y\_pred1"].                                                                                                                                                                  |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **pipe**                       | **contains the sequence of model components (including vocabs, preprocessors, main components, postprocessors etc.).**                                                                                                                                         |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                | **parameters of the main part of a model**                                                                                                                                                                                                                     |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| in                             | the same as "in" parameter in the "chainer".                                                                                                                                                                                                                   |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| in\_y                          | the same as "in\_y" parameter in the "chainer".                                                                                                                                                                                                                |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| out                            | the same as "out" parameter in the "chainer".                                                                                                                                                                                                                  |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| name                           | str, a registered name of the model.                                                                                                                                                                                                                           |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| device\_num                    | int, a GPU card number to train the model on, if several cards are available in a system.                                                                                                                                                                      |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| load\_path                     | str, a path to a file from which model files will be loaded.                                                                                                                                                                                                   |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| save\_path                     | str, a path to a file where model files will be saved.                                                                                                                                                                                                         |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| train\_now                     | bool, if it is True, than the model training will be done, else validation and test only.                                                                                                                                                                      |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| vocabs\_path                   | str, a path to a directory with data files from where the model vocabularies will be built.                                                                                                                                                                    |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| download\_url                  | str, a URL where a pre-trained model with word embeddings is stored.                                                                                                                                                                                           |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| embeddings                     | {"wor2vec", "fasttext"}, a type of the pre-trained embeddings model.                                                                                                                                                                                           |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| embeddings\_path               | str, a path to a folder where to save downloaded embeddings.                                                                                                                                                                                                   |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| embedding\_dim                 | int, a dimensionality of word embeddings vectors.                                                                                                                                                                                                              |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| seed                           | int or None (default=None), a seed to initialize the model weights.                                                                                                                                                                                            |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| max\_sequence\_length          | int, a maximum number of tokens in an input sequence. If the sequence is shorter than the "max\_sequence\_length" it will be padded with a default token, otherwise the sequence will be truncated.                                                            |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| padding                        | {"pre", "post"}, pad either before or after each sequence if it is shorter than "max\_sequence\_length".                                                                                                                                                       |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| truncating                     | {"pre", "post"}, remove values from sequences larger than "max\_sequence\_length", either at the beginning or at the end of the sequences.                                                                                                                     |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| reccurent                      | {"lstm", "bilstm"}, a type of a reccurent neural network (LSTM or bi-LSTM) to encode an input sequence.                                                                                                                                                        |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| max\_pooling                   | bool, if it is True the max-pooling operation will be performed, else the last hidden state from the reccurent neural network will be taken.                                                                                                                   |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| type\_of\_weights              | {"shared", "separate"}, use shared of separate weights to encode the context and response.                                                                                                                                                                     |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| hidden\_dim                    | int, a size of a hidden state if the "reccurent" parameter is set to the "lstm" or the half-size if the "reccurent" is set to the "bilstm".                                                                                                                    |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| learning\_rate                 | float, learning rate for training.                                                                                                                                                                                                                             |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| margin                         | float, a margin to use in a triplet loss.                                                                                                                                                                                                                      |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| load\_path                     | str, a path to a file from which model files will be loaded.                                                                                                                                                                                                   |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| save\_path                     | str, a path to a file where model files will be saved.                                                                                                                                                                                                         |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| interact\_pred\_num            | int, first "interact\_pred\_num" best candidates for context and response to show in the "interact" regime.                                                                                                                                                    |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **train**                      | **parameters for training**                                                                                                                                                                                                                                    |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| epochs                         | int, a number of epochs for training.                                                                                                                                                                                                                          |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| batch\_size                    | int, a batch size for training.                                                                                                                                                                                                                                |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| pytest\_max\_batches           | int, a number of batches for the model training while testing.                                                                                                                                                                                                 |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| metrics                        | a list of metrics names , loss "loss", top-1 recall `"r@1" <mailto:"r@1">`__, `"r@2" <mailto:"r@2">`__, `"r@5 <mailto:"r@5>`__" and the average position of the correct response among all response candidates "rank\_response" are available for the model.   |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| validation\_patience           | int, for how many epochs the training can continue without improvement of the metric value on the validation set.                                                                                                                                              |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| val\_every\_n\_epochs          | int, a frequency of validation during training (validate every n epochs).                                                                                                                                                                                      |
+--------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Comparison
----------

The InsuranceQA V1 dataset:

+------------------------------------------------------------------+-------------------------+--------------------+
| Model                                                            | Validation (Recall@1)   | Test1 (Recall@1)   |
+==================================================================+=========================+====================+
| Architecture II: (HLQA(200) CNNQA(4000) 1-MaxPooling Tanh) [1]   | 61.8                    | 62.8               |
+------------------------------------------------------------------+-------------------------+--------------------+
| QA-LSTM basic-model(max pooling) [2]                             | 64.3                    | 63.1               |
+------------------------------------------------------------------+-------------------------+--------------------+
| Our model (biLSTM, max pooling)                                  | **67.6**                | **67.6**           |
+------------------------------------------------------------------+-------------------------+--------------------+

Literature
----------

[1] Feng, Minwei, et al. "Applying deep learning to answer selection: A
study and an open task." Automatic Speech Recognition and Understanding
(ASRU), 2015 IEEE Workshop on. IEEE, 2015.

[2] Tan, Ming, et al. "LSTM-based deep learning models for non-factoid
answer selection." arXiv preprint arXiv:1511.04108 (2015).
