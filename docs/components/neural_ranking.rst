Neural Model for Information Retrieval and Ranking
==================================================

This library component solves the task of Information Retrieval (IR)
and is intended to retrieve the response closest semantically
to a given context from some database.
The code in this repository uses a deep learning
approach to address the IR task.

The model can be trained with crossentropy or with triplet loss [1,
2]. To train with triplet loss
it is required to provide positive and negative response candidates
for each context.
Sampling of negative candidates can be performed globally from the
whole response set
or from pools of responses predefined separately for each context.
The same is true for validation and test, i.e. validation and test
can be carried out for the entire set of answers or
for the answers pools selected separately for each context.
There is a possibility to encode contexts and responses in the model
with biLSTM layers having shared or separate weights.

Infer from pre-trained model
----------------------------

To use the model pre-trained on the InsuranceQA V1 dataset for
inference, one should run
the following command:

::

    python -m deeppavlov interact deeppavlov/configs/ranking/ranking_insurance.json

Now a user can enter a text of context and get relevant contexts and
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

As an example of configuration file see
:config:`ranking_insurance.json <ranking/ranking_insurance.json>`.

Literature
----------

[1] Feng, Minwei, et al. "Applying deep learning to answer selection: A
study and an open task." Automatic Speech Recognition and Understanding
(ASRU), 2015 IEEE Workshop on. IEEE, 2015.

[2] Tan, Ming, et al. "LSTM-based deep learning models for non-factoid
answer selection." arXiv preprint arXiv:1511.04108 (2015).
