Pre-trained embeddings
============================

ELMo
----

| We are publishing :class:`Russian language ELMo embeddings model <deeppavlov.models.embedders.elmo_embedder.ELMoEmbedder>` for tensorflow-hub.
| ELMo (Embeddings from Language Models) representations are pre-trained contextual representations from
  large-scale bidirectional language models. See a paper `Deep contextualized word representations
  <https://arxiv.org/abs/1802.05365>`__ for more information about the algorithm and a detailed analysis.
| These vectors where trained on `Russian Wikipedia <https://ru.wikipedia.org/>`__.


fastText
--------

We are publishing pre-trained word vectors for Russian language. These
vectors were trained on joint `Russian
Wikipedia <https://ru.wikipedia.org/>`__
and `Lenta.ru <https://lenta.ru/>`__ corpora.

All vectors are 300-dimentional. We used fastText skpip-gram (see
`Bojanowski et al. (2016) <https://arxiv.org/abs/1607.04606>`__) for
vectors training as well as various preprocessing options (see below).

You can get vectors either in binary or in text (vec) formats both for
fastText and GloVe.

License
~~~~~~~

The pre-trained word vectors are distributed under the `License Apache
2.0 <https://www.apache.org/licenses/LICENSE-2.0>`__.

Downloads
~~~~~~~~~

The models can be downloaded from:

+-----------------------+---------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Model                 | Preprocessing                                           | Vectors                                                                                                                                                                                                                                                                                                                            |
+=======================+=========================================================+====================================================================================================================================================================================================================================================================================================================================+
| fastText (skipgram)   | tokenize (nltk word\_tokenize), lemmatize (pymorphy2)   | `bin <http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_lemmatize/ft_native_300_ru_wiki_lenta_lemmatize.bin>`__, `vec <http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_lemmatize/ft_native_300_ru_wiki_lenta_lemmatize.vec>`__                                                                   |
+-----------------------+---------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| fastText (skipgram)   | tokenize (nltk word\_tokenize), lowercasing             | `bin <http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_lower_case/ft_native_300_ru_wiki_lenta_lower_case.bin>`__, `vec <http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_lower_case/ft_native_300_ru_wiki_lenta_lower_case.vec>`__                                                               |
+-----------------------+---------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| fastText (skipgram)   | tokenize (nltk wordpunсt\_tokenize)                     | `bin <http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_nltk_wordpunct_tokenize/ft_native_300_ru_wiki_lenta_nltk_wordpunct_tokenize.bin>`__, `vec <http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_nltk_wordpunct_tokenize/ft_native_300_ru_wiki_lenta_nltk_wordpunct_tokenize.vec>`__           |
+-----------------------+---------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| fastText (skipgram)   | tokenize (nltk word\_tokenize)                          | `bin <http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize/ft_native_300_ru_wiki_lenta_nltk_word_tokenize.bin>`__, `vec <http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize/ft_native_300_ru_wiki_lenta_nltk_word_tokenize.vec>`__                               |
+-----------------------+---------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| fastText (skipgram)   | tokenize (nltk word\_tokenize), remove stopwords        | `bin <http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_remstopwords/ft_native_300_ru_wiki_lenta_remstopwords.bin>`__, `vec <http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_remstopwords/ft_native_300_ru_wiki_lenta_remstopwords.vec>`__                                                       |
+-----------------------+---------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Word vectors training parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These word vectors were trained with following parameters ([...] is for
default value):

fastText (skipgram)
                   

-  lr [0.1]
-  lrUpdateRate [100]
-  dim 300
-  ws [5]
-  epoch [5]
-  neg [5]
-  loss [softmax]
-  pretrainedVectors []
-  saveOutput [0]

