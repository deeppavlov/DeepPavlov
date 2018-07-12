# TFIDF ranker

This is an implementation of a document ranker based on tfidf vectorization. The ranker implementation
is based on [DrQA](https://github.com/facebookresearch/DrQA) project.


### Config

Default ranker config for **English** language is `deeppavlov/configs/ranking/en_ranker_tfidf_wiki.json`

Default ranker config for **Russian** language is `deeppavlov/configs/ranking/ru_ranker_tfidf_wiki.json`

The ranker config for **English** language can be found at `deeppavlov/configs/odqa/en_ranker_prod.json`

The ranker config for **Russian** language can be found at `deeppavlov/configs/odqa/ru_ranker_prod.json`

* **dataset_iterator** - downloads Wikipidia DB, creates batches for ranker fitting
    * **_data_dir_** - a directory to download DB to
    * **_data_url_** - an URL to download Wikipedia DB from
    * **_shuffle_** - whether to perform shuffling when iterating over DB or not
* **chainer** - pipeline manager
    * **_in_** - pipeline input data (questions)
    * **_out_** - pipeline output data (Wikipedia articles ids and scores of the articles)
* **tfidf_ranker** - the ranker class
    * **top_n** - a number of document to return (when n=1 the most relevant document is returned)
    * **_in_** - ranker input data (questions)
    * **_out_** - ranker output data (Wikipedia articles ids)
    * **_fit_on_batch_** - fit the ranker on batches of Wikipedia articles
    * **_vectorizer_** - a vectorizer class
        * **_fit_on_batch_** - fit the vectorizer on batches of Wikipedia articles
        * **_save_path_** - a path to serialize a vectorizer to
        * **_load_path_** - a path to load a vectorizer from
        * **_tokenizer_** - a tokenizer class
            * **_lemmas_** - whether to lemmatize tokens or not
            * **_ngram_range_** - ngram range for vectorizer features
* **train** - parameters for vectorizer fitting
    * **_validate_best_**- is ingnored, any value
    * **_test_best_** - is ignored, any value
    * **_batch_size_** - how many Wikipedia articles should return the dataset iterator in a single batch

## Running the ranker

**Training and infering the rannker on English Wikipedia requires 16 GB RAM**

## Training

Run the following to fit the ranker on **English** Wikipedia:
```bash
cd deeppavlov/
python deep.py train deeppavlov/configs/ranking/en_ranker_tfidf_wiki.json
```
Run the following to fit the ranker on **Russian** Wikipedia:
```bash
cd deeppavlov/
python deep.py train deeppavlov/configs/ranking/ru_ranker_tfidf_wiki.json
```

## Interacting

When interacted the ranker returns document titles of the relevant documents.

Run the following to interact the **English** ranker:
```bash
cd deeppavlov/
python deep.py interact deeppavlov/configs/ranking/en_ranker_tfidf_wiki.json -d
```
Run the following to interact the **Russian** ranker:
```bash
cd deeppavlov/
python deep.py interact deeppavlov/configs/ranking/ru_ranker_tfidf_wiki.json -d
```

## Pretrained models

Wikipedia DB and pretrained tfidf matrices are downloaded in `deeppavlov/download/odqa` by default.

### enwiki.db

**enwiki.db** SQLite database consists of **5159530 Wikipedia articles**
and is built by the following steps:
1. Download a Wikipedia dump file. We took the latest [enwiki](https://dumps.wikimedia.org/enwiki/20180201)
 (from 2018-02-11)
2. Unpack and extract the articles with [WikiExtractor](https://github.com/attardi/wikiextractor)
 (with `--json`, `--no-templates`, `--filter_disambig_pages` options)
3. Build a database with the help of [DrQA script](https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py).

### enwiki_tfidf_matrix.npz

 **enwiki_tfidf_matrix.npz** is a full Wikipedia tf-idf matrix of size `hash_size x number of documents` which is
 `2**24 x 5159530`. This matrix is built with `deeppavlov/models/vectorizers/hashing_tfidf_vectorizer.HashingTfidfVectorizer`
 class.

### ruwiki.db

**ruwiki.db** SQLite database consists of **1463888 Wikipedia articles**
and is built by the following steps:
1. Download a Wikipedia dump file. We took the latest [ruwiki](https://dumps.wikimedia.org/ruwiki/20180401)
(from 2018-04-01)
2. Unpack and extract the articles with [WikiExtractor](https://github.com/attardi/wikiextractor)
(with `--json`, `--no-templates`, `--filter_disambig_pages` options)
3. Build a database with the help of [DrQA script](https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py).

### ruwiki_tfidf_matrix.npz

 **ruwiki_tfidf_matrix.npz** is a full Wikipedia tf-idf matrix of size `hash_size x number of documents` which is
 `2**24 x 1463888`. This matrix is built with `deeppavlov/models/vectorizers/hashing_tfidf_vectorizer.HashingTfidfVectorizer`
 class.


## References

1. https://github.com/facebookresearch/DrQA