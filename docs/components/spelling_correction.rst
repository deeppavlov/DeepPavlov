Automatic spelling correction pipelines
=======================================

We provide two types of pipelines for spelling correction:
`levenshtein_corrector <#levenshtein-corrector>`__
uses simple Damerau-Levenshtein distance to find correction candidates
and `brillmoore <#brillmoore>`__
uses statistics based error model for it. In both cases correction
candidates are chosen based on context
with the help of a `kenlm language model <#language-model>`__.
You can find `the comparison <#comparison>`__ of these and other
approaches near the end of this readme.

.. note::

    About 4.4 GB on disc required for the Russian language model and about 7 GB for the English one.

Quick start
-----------

First you would need to install additional requirements:

::

    python -m deeppavlov install <path_to_config>

where ``<path_to_config>`` is a path to one of the :config:`provided config files <spelling_correction>`
or its name without an extension, for example :config:`levenshtein_corrector_ru <spelling_correction/levenshtein_corrector_ru.json>`.

You can run the following command to try provided pipelines out:

::

    python -m deeppavlov interact <path_to_config> [-d]

where ``<path_to_config>`` is one of the :config:`provided config files <spelling_correction>`.
With the optional ``-d`` parameter all the data required to run
selected pipeline will be downloaded, including
an appropriate language model.

After downloading the required files you can use these configs in your
python code.
For example, this code will read lines from stdin and print corrected
lines to stdout:

.. code:: python

    import sys

    from deeppavlov import build_model, configs

    CONFIG_PATH = configs.spelling_correction.brillmoore_kartaslov_ru

    model = build_model(CONFIG_PATH, download=True)
    for line in sys.stdin:
        print(model([line])[0], flush=True)

levenshtein_corrector
---------------------

:class:`This component <deeppavlov.models.spelling_correction.levenshtein.LevenshteinSearcherComponent>` finds all the
candidates in a static dictionary on a set Damerau-Levenshtein distance.
It can separate one token into two but it will not work the other way
around.

Component config parameters:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  ``in`` — list with one element: name of this component's input in
   chainer's shared memory
-  ``out`` — list with one element: name for this component's output in
   chainer's shared memory
-  ``class_name`` always equals to ``"spelling_levenshtein"`` or ``deeppavlov.models.spelling_correction.levenshtein.searcher_component:LevenshteinSearcherComponent``.
-  ``words`` — list of all correct words (should be a reference)
-  ``max_distance`` — maximum allowed Damerau-Levenshtein distance
   between source words and candidates
-  ``error_probability`` — assigned probability for every edit

brillmoore
----------

:class:`This component <deeppavlov.models.spelling_correction.brillmoore.ErrorModel>` is based on
`An Improved Error Model for Noisy Channel Spelling Correction <http://www.aclweb.org/anthology/P00-1037>`_
by Eric Brill and Robert C. Moore and uses statistics based error
model to find best candidates in a static dictionary.

Component config parameters:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  ``in`` — list with one element: name of this component's input in
   chainer's shared memory
-  ``out`` — list with one element: name for this component's output in
   chainer's shared memory
-  ``class_name`` always equals to ``"spelling_error_model"`` or ``deeppavlov.models.spelling_correction.brillmoore.error_model:ErrorModel``.
-  ``save_path`` — path where the model will be saved at after a
   training session
-  ``load_path`` — path to the pretrained model
-  ``window`` — window size for the error model from ``0`` to ``4``,
   defaults to ``1``
-  ``candidates_count`` — maximum allowed count of candidates for every
   source token
-  ``dictionary`` — description of a static dictionary model, instance
   of (or inherited from)
   ``deeppavlov.vocabs.static_dictionary.StaticDictionary``

   -  ``class_name`` — ``"static_dictionary"`` for a custom dictionary or one
      of two provided:

      -  ``"russian_words_vocab"`` to automatically download and use a
         list of russian words from
         `https://github.com/danakt/russian-words/ <https://github.com/danakt/russian-words/>`__
      -  ``"wikitionary_100K_vocab"`` to automatically download a list
         of most common words from Project Gutenberg from
         `Wiktionary <https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists#Project_Gutenberg>`__

   -  ``dictionary_name`` — name of a directory where a dictionary will
      be built to and loaded from, defaults to ``"dictionary"`` for
      static\_dictionary
   -  ``raw_dictionary_path`` — path to a file with a line-separated
      list of dictionary words, required for static\_dictionary

Training configuration
^^^^^^^^^^^^^^^^^^^^^^

For the training phase config file needs to also include these
parameters:

-  ``dataset_iterator`` — it should always be set like
   ``"dataset_iterator": {"class_name": "typos_iterator"}``

   -  ``class_name`` always equals to ``typos_iterator``
   -  ``test_ratio`` — ratio of test data to train, from ``0.`` to
      ``1.``, defaults to ``0.``

-  ``dataset_reader``

   -  ``class_name`` — ``typos_custom_reader`` for a custom dataset or one of
      two provided:

      -  ``typos_kartaslov_reader`` to automatically download and
         process misspellings dataset for russian language from
         https://github.com/dkulagin/kartaslov/tree/master/dataset/orfo_and_typos
      -  ``typos_wikipedia_reader`` to automatically download and
         process a list of common misspellings from english
         Wikipedia - https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines

   -  ``data_path`` — required for typos\_custom\_reader as a path to
      a dataset file,
      where each line contains a misspelling and a correct spelling
      of a word separated by a tab symbol

Component's configuration for ``spelling_error_model`` also has to
have as ``fit_on`` parameter — list of two elements:
names of component's input and true output in chainer's shared
memory.

Language model
--------------

Provided pipelines use `KenLM <http://kheafield.com/code/kenlm/>`__ to
process language models, so if you want to build your own,
we suggest you consult its website. We do also provide our own
language models for
`english <http://files.deeppavlov.ai/lang_models/en_wiki_no_punkt.arpa.binary.gz>`__
(5.5GB) and
`russian <http://files.deeppavlov.ai/lang_models/ru_wiyalen_no_punkt.arpa.binary.gz>`__
(3.1GB) languages.

Comparison
----------

We compared our pipelines with
`Yandex.Speller <http://api.yandex.ru/speller/>`__,
`JamSpell <https://github.com/bakwc/JamSpell>`__  and
`PyHunSpell <https://github.com/blatinier/pyhunspell>`__
on the `test set <http://www.dialog-21.ru/media/3838/test_sample_testset.txt>`__
for the `SpellRuEval
competition <http://www.dialog-21.ru/en/evaluation/2016/spelling_correction/>`__
on Automatic Spelling Correction for Russian:

+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+
| Correction method                                                                       | Precision | Recall | F-measure | Speed (sentences/s) |
+=========================================================================================+===========+========+===========+=====================+
| Yandex.Speller                                                                          | 83.09     | 59.86  | 69.59     | 5.                  |
+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+
| :config:`Damerau Levenshtein 1 + lm<spelling_correction/levenshtein_corrector_ru.json>` | 59.38     | 53.44  | 56.25     | 39.3                |
+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+
| :config:`Brill Moore top 4 + lm<spelling_correction/brillmoore_kartaslov_ru.json>`      | 51.92     | 53.94  | 52.91     | 0.6                 |
+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+
| Hunspell + lm                                                                           | 41.03     | 48.89  | 44.61     | 2.1                 |
+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+
| JamSpell                                                                                | 44.57     | 35.69  | 39.64     | 136.2               |
+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+
| :config:`Brill Moore top 1 <spelling_correction/brillmoore_kartaslov_ru_nolm.json>`     | 41.29     | 37.26  | 39.17     | 2.4                 |
+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+
| Hunspell                                                                                | 30.30     | 34.02  | 32.06     | 20.3                |
+-----------------------------------------------------------------------------------------+-----------+--------+-----------+---------------------+

