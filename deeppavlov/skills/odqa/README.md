# Open Domain Question Answering Skill on Wikipedia

## Task definition

Open Domain Question Answering (ODQA) is a task to find an exact answer to any question in
Wikipedia articles. Thus, given only a question, the system outputs the best answer it can find:

Question:
> What is the name of Darth Vader's son?

Answer:
> Luke Skywalker

## Languages

There are pretrained ODQA models for **English** and **Russian** languages in DeepPavlov.

## Models

The architecture of ODQA skill is modular and consists of two models, a ranker and a reader. The ranker is based on
DrQa proposed by Facebook Research ([Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051))
and the reader is based on R-Net proposed by Microsoft Research Asia (["R-NET: Machine Reading Comprehension with Self-matching Networks"](https://www.microsoft.com/en-us/research/publication/mrc/))
and its [implementation](https://github.com/HKUST-KnowComp/R-Net) by Wenxuan Zhou.

## Running ODQA

**Tensorflow-1.4.0 with GPU support is required** to run this model.

## Training

The ODQA ranker and ODQA reader should be trained separately.
**Warning: training the ranker on English Wikipedia requires 16 GB RAM.** Run the following to fit the ranker:
```bash
python -m deeppavlov.deep train deeppavlov/configs/odqa/en_ranker_prod.json
```
Read about training the reader in our separate [reader tutorial](https://github.com/deepmipt/DeepPavlov/tree/master/deeppavlov/models/squad).

## Interacting

ODQA, reader and ranker can be interacted separately. **Warning: interacting the ranker and ODQA on English Wikipedia requires 16 GB RAM.**
Run the following to interact ODQA:
```bash
python -m deeppavlov.deep train deeppavlov/configs/odqa/en_odqa_infer_prod.json
```
Run the following to interact the ranker:
```bash
python -m deeppavlov.deep interact deeppavlov/configs/odqa/en_ranker_prod.json
```
Read about interacting the reader in our separate [reader tutorial](https://github.com/deepmipt/DeepPavlov/tree/master/deeppavlov/models/squad).

## Configuration

The ODQA configs suit only model inferring purposes. The [ranker config](#the-ranker-config) should be used for ranker training
and the [reader config](https://github.com/deepmipt/DeepPavlov/tree/master/deeppavlov/models/squad#config-components) should be used for reader training.

### Ranker

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

### ODQA

Default ODQA config for **English** language is `deeppavlov/configs/odqa/en_odqa_infer_prod.json`

Default ODQA config for **Russian** language is `deeppavlov/configs/odqa/ru_odqa_infer_prod.json`

The components of ODQA config can be referred to [ranker config](#the-ranker-config)
and [reader config](https://github.com/deepmipt/DeepPavlov/tree/master/deeppavlov/models/squad#config-components) accordingly.
However, main inputs and outputs are worth explaining:

* **chainer** - pipeline manager
    * **_in_** - pipeline input data (questions)
    * **_out_** - pipeline output data (answers)

## Pretrained models

Wikipedia data and pretrained ODQA models are downloaded in `deeppavlov/download/odqa` by default.

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
2. https://github.com/HKUST-KnowComp/R-Net