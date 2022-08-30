Generative Question Answering
=============================

Task definitfion
----------------
Generative Question Answering is the task of finding an answer on question in a given contexts (e.g, paragraphs from Wikipedia),
where the answer to each question is **not necessary** a segment of the context.


**Question**:

    Is it possible to have a rating above 4000 in chess?

**Contexts**:

    > Right now that can't really happen. Now, the highest-rated chess player is Stockfish 12, with a rating of 3515. A rating difference of 400 points means you'll beat your opponent over 90% of the time. Here we're looking at an even bigger difference than that: about 500 points.

    > It's nearly impossible to measure the rating difference between two players so far apart in skill. For there to be a player with a rating of 4000, there would first have to be other players with ratings that are at least fairly close, like 3800.

**Answer**:
    
    not really possible

Datasets
--------
We consider the following datasets:

-  `Natural Questions <https://ai.google.com/research/NaturalQuestions>`__
-  `TriviaQA <https://nlp.cs.washington.edu/triviaqa>`__

Specifically, we validate our model on *Natural Questions* and *TriviaQA* from: https://github.com/facebookresearch/FiD.


Datasets format
~~~~~~~~~~~~~~~
{
    "train":    [ 
                    [ 
                        [ "question", [ "contexts" ], [ "titles" ] ],

                        [ "target", [ "answers" ] ]

                    ],
                    
                    ...

                ]

    "valid":    [ ... ]

    "test":     [ ... ]

}

Built-In Models
---------------
DeepPavlov's model for generative question answering is based on Fusion-in-decoder(FiD) base.
The model generates answer based on the question and k-support contexts.

Currently, we provide two built-in models for generative question answering in DeepPavlov library, finetuned on 2 datasets:

-  Natural Questions :config:`deeppavlov/configs/generative_qa/nq_fid.json`

-  TriviaQA :config:`deeppavlov/configs/generative_qa/tqa_fid.json`

Architecture
~~~~~~~~~~~~
FiD model uses several support passages to gather usefull information from multiple knowledge sources. Firstly, every 
passage is concatinated with the question like this *"question: What is the capital of UK? passage: London is the capital of UK"*
and processed independently from other passages by the encoder of pretrained sequence-to-sequence network (e.g. T5). 
Then the decoder performs attention over the concatenation of the resulting representations of all the retrieved passages


Metrics
~~~~~~~
Natural Questions dataset
^^^^^^^^^^^^^^^^^^^^^^^^^
+---------------------------------------------------------+---------------------------------+---------------------------------+
|                                                 Dataset |      Natural Questions (dev)    |      Natural Questions (test)   |
+---------------------------------------------------------+----------------+----------------+----------------+----------------+
| Model                                                   |       EM       |       F-1      |       EM       |       F-1      |
+=========================================================+================+================+================+================+
| :config:`DeepPavlov FiD <generative_qa/nq_fid.json>`    |      39.9      |      50.0      |      46.0      |      54.1      |
+---------------------------------------------------------+----------------+----------------+----------------+----------------+
| `T5`_                                                   |      42.0      |      50.6      |      42.2      |      49.7      |
+---------------------------------------------------------+----------------+----------------+----------------+----------------+


TriviaQA dataset
^^^^^^^^^^^^^^^^
+---------------------------------------------------------+---------------------------------+---------------------------------+
|                                                 Dataset |          TriviaQA (dev)         |          TriviaQA (test)        |         
+---------------------------------------------------------+----------------+----------------+----------------+----------------+
| Model                                                   |       EM       |       F-1      |       EM       |       F-1      |
+=========================================================+================+================+================+================+
| :config:`DeepPavlov FiD <generative_qa/tqa_fid.json>`   |      61.8      |      69.6      |      63.1      |      70.0      |
+---------------------------------------------------------+----------------+----------------+----------------+----------------+
| :config:`DeepPavlov FiD <generative_qa/nq_fid.json>`    |      51.1      |      61.3      |      52.2      |      61.9      |
+---------------------------------------------------------+----------------+----------------+----------------+----------------+
| `T5`_                                                   |      46.0      |      55.0      |      46.1      |      55.3      |
+---------------------------------------------------------+----------------+----------------+----------------+----------------+
| `QANet`_                                                |      51.1      |      56.6      |       --       |       --       |
+---------------------------------------------------------+----------------+----------------+----------------+----------------+
| `M-Reader`_                                             |       --       |       --       |      46.9      |      52.9      |
+---------------------------------------------------------+----------------+----------------+----------------+----------------+
| `MEMEN`_                                                |      43.2      |      46.9      |       --       |       --       |
+---------------------------------------------------------+----------------+----------------+----------------+----------------+
| `BiDAF`_                                                |      40.3      |      45.7      |       --       |       --       |
+---------------------------------------------------------+----------------+----------------+----------------+----------------+


.. _`M-Reader`: https://arxiv.org/abs/1705.02798
.. _`MEMEN`: https://arxiv.org/abs/1707.09098
.. _`QANet`: https://arxiv.org/abs/1804.09541
.. _`BiDAF`: https://arxiv.org/abs/1611.01603
.. _`T5`: https://arxiv.org/abs/1910.10683



Prerequisites
-------------

Before using the models make sure that all required packages are installed running the command:

    .. code:: bash

        python -m deeppavlov install nq_fid
        python -m deeppavlov install tqa_fid


Pretrained models are available and can be downloaded (~0.9Gb):

    .. code:: bash

        python -m deeppavlov download nq_fid
        python -m deeppavlov download tqa_fid


Model usage from Python
-----------------------

Interact
~~~~~~~~
    .. code:: python

        from deeppavlov import build_model

        model = build_model('nq_fid', download=True)

        model([
                "What is the capital of UK?",
                "Where did the name Atari itself come from?"
            ],
            [
                [
                    "The name Britain is sometimes used to refer to the United Kingdom as a whole",
                    "London is the capital of Great Britain"
                ],
                [
                    "Bushnell and Dabney were originally going to name their company Syzygy, a term for planetary alignment, but found that it had been registered already.",
                    "Instead, they chose a word from the Japanese game Go. The Japanese equivalent of chess, in Go Atari means something similar to \'check\'."
                ]
            ])
        >>> ['london', 'the japanese game go']

        model([   
                "How many points do you need to win in badminton?"
            ],
            [
                [
                "A rally is lost if the shuttle is hit into the net, or over the net but outside of the opponent's court.",
                "A rally is also lost if the shuttle touches the player's clothing or body, or if it is hit before it crosses over the net",
                'The side winning a rally adds a point to its score', 'A match consists of the best of 3 games of 21 points (games cap at 30 points)',
                "A rally is won when a shuttle is hit over the net and onto the floor of the opponent's court.",
                'At 29 all, the side scoring the 30th point, wins that game',
                'The side winning a game serves first in the next game',
                'At 20 all, the side which gains a 2 point lead first, wins that game.',
                'Each gamestarts at 0-0. If the match goes to the third game that third game will be played to 15'
                ]
            ])
        >>> ['21']

Train
~~~~~
    .. code:: python

        from deeppavlov import train_model

        model = train_model('nq_fid', download=True)


Model usage from CLI
--------------------

Train
~~~~~
    .. code:: bash

        python -m deeppavlov train nq_fid

Evaluate
~~~~~~~~
    .. code:: bash

        python -m deeppavlov evaluate nq_fid

Interact
~~~~~~~~

Interact mode provides command line interface to already trained model.

    .. code:: bash

        python -m deeppavlov interact nq_fid
