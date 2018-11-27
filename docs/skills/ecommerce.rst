eCommerce Bot
======================

The eCommerce bot helps you to identify the most relevant product items according to your search query. The retrieval is based on the list of the ranking measures. In addition, when retrieved candidates are not specific enough, the bot asks you to provide additional information to specify the search (as on the example below).

Here is a simple example of interaction:

.. code:: bash

	>>	Hello, I am a new eCommerce bot. 
		I will help you to find products that you are looking for. 
		Please type your query in plain text.

	x::bluetooth speaker
	>>	This is what I found for you:

		- Bluetooth Speaker (Black & Red)
		- Bose SoundLink Bluetooth Speaker III
		- Bose SoundLink Mini Bluetooth Speaker
		- Bose SoundLink Mini Bluetooth Speaker
		- JBL Flip Wireless Bluetooth Speaker (Black)

		To specify the search, please choose a Brand: JBL, Soundsworks
	
	x::JBL
	>>	The following items satisfy your request
		
		- JBL Flip Wireless Bluetooth Speaker (Black)
		- JBL Flip Wireless Bluetooth Speaker (Black)
		- JBL Charge Portable Indoor/Outdoor Bluetooth Speaker | Black


Usage
-----

Config file
^^^^^^^^^^^

BLEU-based `[1] <#references>`__ eCommerce bot 
:config:`ecommerce_skill/bleu_retrieve.json`

TfIdf-based eCommerce bot 
:config:`ecommerce_skill/tfidf_retrieve.json`


Usage example
^^^^^^^^^^^^^

To interact with a pretrained model run:

.. code:: bash

    python -m deeppavlov interact <path_to_config> [-d]

where ``<path_to_config>`` is a path to config file.

You can also train your own model by specifying config file and running:

.. code:: bash

    python -m deeppavlov train <path_to_config>

Configuration settings
----------------------

The eCommerce bot configuration consists of the following parts:

-  **dataset_reader**
-  **dataset_iterator**
-  **chainer**

You can use your own **dataset_reader**, **dataset_iterator** for specific data.

eCommerce bot with BLEU-based ranker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **chainer** - pipeline manager

   -  ``in`` - pipeline input data: an user ``query``, a dialog ``state`` and dialog history ``history``.
   -  ``out`` - pipeline output data: ``response`` the structure with retrieved product items.

-  **ecommerce_skill_bleu** - ranker 

   -  ``min_similarity``: lower boundary for textual similarity ranker (by default 0.5).
   -  ``min_entropy``: lower boundary for entropy (by default 0.5). If the entropy is less than ``min_entropy``, it's omitted from the specification list.
   -  ``entropy_fields``: the specification attributes of the catalog items (by default "Size", "Brand", "Author", "Color", "Genre").
   -  ``preprocess``: text preprocessing component.


   **Input:**

   -  ``query``: a plain text user query.
   -  ``history``: dialog history.
   -  ``state``: dialog state.


   **Returns:**

   -  ``items``: product items in sorted order from ``start`` index till ``end`` index (taken from the dialog state).
   -  ``entropies``: specification attributes with corresponding values in sorted order.
   -  ``confidence``: similarity confidence.
   -  ``state``: dialog state.


   .. note::

      About 500 Mb on disc required for eCommerce bot with BLEU-based ranker.


eCommerce bot with TfIdf-based ranker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **chainer** - pipeline manager

   -  ``in`` - pipeline input data: an user ``query``, a dialog ``state`` and dialog history ``history``.
   -  ``out`` - pipeline output data: ``response`` the structure with retrieved product items.

-  **ecommerce_skill_tfidf** - ranker 

   -  ``min_similarity``: lower boundary for textual similarity ranker (by default 0.5).
   -  ``min_entropy``: lower boundary for entropy (by default 0.5). If the entropy is less than ``min_entropy``, it's omitted from the specification list.
   -  ``entropy_fields``: the specification attributes of the catalog items (by default "Size", "Brand", "Author", "Color", "Genre").


   **Input:**

   -  ``query``: a plain text user query.
   -  ``history``: dialog history.
   -  ``state``: dialog state.


   **Returns:**

   -  ``items``: product items in sorted order from ``start`` index till ``end`` index (taken from the dialog state).
   -  ``entropies``: specification attributes with corresponding values in sorted order.
   -  ``confidence``: similarity confidence.
   -  ``state``: dialog state.


   .. note::

      About 130 Mb on disc required for eCommerce bot with TfIdf-based ranker


References
----------

[1]  Papineni, Kishore, et al. "BLEU: a method for automatic evaluation 
of machine translation." Proceedings of the 40th annual meeting on association 
for computational linguistics. Association for Computational Linguistics, 2002.
