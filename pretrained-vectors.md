[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](/LICENSE.txt)
[![fastText](https://img.shields.io/badge/word%20vectors-fastText-brightgreen.svg)](https://fasttext.cc/)
[![GloVe](https://img.shields.io/badge/word%20vectors-GloVe-brightgreen.svg)](https://nlp.stanford.edu/projects/glove/)


# Pre-trained word vectors

We are publishing pre-trained word vectors for Russian language. These vectors were trained on joint [*Russian Wikipedia*](https://ru.wikipedia.org/wiki/%D0%97%D0%B0%D0%B3%D0%BB%D0%B0%D0%B2%D0%BD%D0%B0%D1%8F_%D1%81%D1%82%D1%80%D0%B0%D0%BD%D0%B8%D1%86%D0%B0) and [*Lenta.ru*](https://lenta.ru/) corpora. 

All vectors are 300-dimentional. We used fastText skpip-gram (see [*Bojanowski et al. (2016)*](https://arxiv.org/abs/1607.04606)) and GloVe (see [*Pennington et al. (2014)*](https://nlp.stanford.edu/pubs/glove.pdf)) for vectors training as well as various preprocessing options (see below).

You can get vectors either in binary or in text (vec) formats both for fastText and GloVe.

## License

The pre-trained word vectors are distributed under the [*License Apache 2.0*](https://www.apache.org/licenses/LICENSE-2.0).

## Downloads

The models can be downloaded from:

| Model               | Preprocessing                                        | Vectors                                                                                                                                                                                                                                                                                                                      |
|---------------------|------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| fastText (skipgram) | tokenize (nltk word_tokenize), lemmatize (pymorphy2) | [*bin*](http://lnsigo.mipt.ru/export/embeddings/ft_native_300_ru_wiki_lenta_lemmatize/ft_native_300_ru_wiki_lenta_lemmatize.bin),                               [*vec*](http://lnsigo.mipt.ru/export/embeddings/ft_native_300_ru_wiki_lenta_lemmatize/ft_native_300_ru_wiki_lenta_lemmatize.vec)                             |
| fastText (skipgram) | tokenize (nltk word_tokenize), lowercasing           | [*bin*](http://lnsigo.mipt.ru/export/embeddings/ft_native_300_ru_wiki_lenta_lower_case/ft_native_300_ru_wiki_lenta_lower_case.bin),                             [*vec*](http://lnsigo.mipt.ru/export/embeddings/ft_native_300_ru_wiki_lenta_lower_case/ft_native_300_ru_wiki_lenta_lower_case.vec)                           |
| fastText (skipgram) | tokenize (nltk wordpunсt_tokenize)                   | [*bin*](http://lnsigo.mipt.ru/export/embeddings/ft_native_300_ru_wiki_lenta_nltk_wordpunct_tokenize/ft_native_300_ru_wiki_lenta_nltk_wordpunct_tokenize.bin),   [*vec*](http://lnsigo.mipt.ru/export/embeddings/ft_native_300_ru_wiki_lenta_nltk_wordpunct_tokenize/ft_native_300_ru_wiki_lenta_nltk_wordpunct_tokenize.vec) |
| fastText (skipgram) | tokenize (nltk word_tokenize)                        | [*bin*](http://lnsigo.mipt.ru/export/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize/ft_native_300_ru_wiki_lenta_nltk_word_tokenize.bin),             [*vec*](http://lnsigo.mipt.ru/export/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize/ft_native_300_ru_wiki_lenta_nltk_word_tokenize.vec)           |
| fastText (skipgram) | tokenize (nltk word_tokenize), remove stopwords      | [*bin*](http://lnsigo.mipt.ru/export/embeddings/ft_native_300_ru_wiki_lenta_remstopwords/ft_native_300_ru_wiki_lenta_remstopwords.bin),                         [*vec*](http://lnsigo.mipt.ru/export/embeddings/ft_native_300_ru_wiki_lenta_remstopwords/ft_native_300_ru_wiki_lenta_remstopwords.vec)                       |
| GloVe               | tokenize (nltk word_tokenize), lemmatize (pymorphy2) | [*bin*](http://lnsigo.mipt.ru/export/embeddings/glove_300_ru_wiki_lenta_lemmatize/glove_300_ru_wiki_lenta_lemmatize.bin),                                       [*text*](http://lnsigo.mipt.ru/export/embeddings/glove_300_ru_wiki_lenta_lemmatize/glove_300_ru_wiki_lenta_lemmatize.txt)                                    |
| GloVe               | tokenize (nltk word_tokenize), lowercasing           | [*bin*](http://lnsigo.mipt.ru/export/embeddings/glove_300_ru_wiki_lenta_lower_case/glove_300_ru_wiki_lenta_lower_case.bin),                                     [*text*](http://lnsigo.mipt.ru/export/embeddings/glove_300_ru_wiki_lenta_lower_case/glove_300_ru_wiki_lenta_lower_case.txt)                                  |
| GloVe               | tokenize (nltk wordpunсt_tokenize)                   | [*bin*](http://lnsigo.mipt.ru/export/embeddings/glove_300_ru_wiki_lenta_nltk_wordpunct_tokenize/glove_300_ru_wiki_lenta_nltk_wordpunct_tokenize.bin),           [*text*](http://lnsigo.mipt.ru/export/embeddings/glove_300_ru_wiki_lenta_nltk_wordpunct_tokenize/glove_300_ru_wiki_lenta_nltk_wordpunct_tokenize.txt)        |
| GloVe               | tokenize (nltk word_tokenize)                        | [*bin*](http://lnsigo.mipt.ru/export/embeddings/glove_300_ru_wiki_lenta_nltk_word_tokenize/glove_300_ru_wiki_lenta_nltk_word_tokenize.bin),                     [*text*](http://lnsigo.mipt.ru/export/embeddings/glove_300_ru_wiki_lenta_nltk_word_tokenize/glove_300_ru_wiki_lenta_nltk_word_tokenize.txt)                  |
| GloVe               | tokenize (nltk word_tokenize), remove stopwords      | [*bin*](http://lnsigo.mipt.ru/export/embeddings/glove_300_ru_wiki_lenta_remstopwords/glove_300_ru_wiki_lenta_remstopwords.bin),                                 [*text*](http://lnsigo.mipt.ru/export/embeddings/glove_300_ru_wiki_lenta_remstopwords/glove_300_ru_wiki_lenta_remstopwords.txt)                              |

## Word vectors training parameters

These word vectors were trained with following parameters ([...] is for default value):

###### fastText (skipgram)
* lr [0.1]
* lrUpdateRate [100]
* dim 300
* ws [5]
* epoch [5]
* neg [5]
* loss [softmax]
* pretrainedVectors []
* saveOutput [0]

###### GloVe
* VERBOSE [2]
* MEMORY [4.0]
* VOCAB_MIN_COUNT [5]
* VECTOR_SIZE 300
* MAX_ITER [15]
* WINDOW_SIZE [15]
* BINARY [2]
* NUM_THREADS 8
* X_MAX [10]