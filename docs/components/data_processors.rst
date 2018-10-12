Data processors
===============

Preprocessors
-------------

Preprocessor is a component that processes batch of samples.

* Already implemented universal preprocessors of **tokenized texts** (each sample is a list of tokens):

    - **CharSplitter** (registered as ``char_splitter``) splits every token in given batch of tokenized samples to a sequence of characters.

    - **Mask** (registered as ``mask``) returns binary mask of corresponding length (padding up to the maximum length per batch.

    - **PymorphyRussianLemmatizer** (registered as ``pymorphy_russian_lemmatizer``) performs lemmatization  for Russian language.

    - **Sanitizer** (registered as ``sanitizer``) removes all combining characters like diacritical marks from tokens.


* Already implemented universal preprocessors of **non-tokenized texts** (each sample is a string):

    - **DirtyCommentsPreprocessor** (registered as ``dirty_comments_preprocessor``) preprocesses samples converting samples to lowercase, paraphrasing English combinations with apostrophe ``'``,  transforming more than three the same symbols to two symbols.

    - **StrLower** (registered as ``str_lower``) converts samples to lowercase.


* Already implemented universal preprocessors of another type of features:

    - **OneHotter** (registered as ``one_hotter``) performs one-hotting operation for the batch of samples where each sample is an integer label or a list of integer labels (can be combined in one batch). If ``multi_label`` parameter is set to ``True``, returns one one-dimensional vector per sample with several elements equal to ``1``.


Tokenizers
----------

Tokenizer is a component that processes batch of samples (each sample is a text string).

    - **LazyTokenizer** (registered as ``lazy_tokenizer``) tokenizes using ``nltk.word_tokenize``.

    - **NLTKTokenizer** (registered as ``nltk_tokenizer``) tokenizes using tokenizers from ``nltk.tokenize``, e.g. ``nltk.tokenize.wordpunct_tokenize``.

    - **NLTKMosesTokenizer** (registered as ``nltk_moses_tokenizer``) tokenizes and detokenizes using ``nltk.tokenize.moses.MosesDetokenizer``, ``nltk.tokenize.moses.MosesTokenizer``.

    - **RuSentTokenizer** (registered as  ``ru_sent_tokenizer``) is a rule-based tokenizer for Russian language.

    - **RussianTokenizer** (registered as ``ru_tokenizer``) tokenizes or lemmatizes Russian texts using ``nltk.tokenize.toktok.ToktokTokenizer``.

    - **StreamSpacyTokenizer** (registered as ``stream_spacy_tokenizer``) tokenizes or lemmatizes texts with spacy ``en_core_web_sm`` models by default.

    - **SplitTokenizer** (registered as ``split_tokenizer``) tokenizes using string method ``split``.


Embedders
---------

Embedder is a component that converts every token in a tokenized batch to a vector of particular dimensionality (optionally, returns a single vector per sample).

    - **GloVeEmbedder** (registered as ``glove``) reads embedding file in GloVe format (file starts with ``number_of_words embeddings_dim line`` followed by lines ``word embedding_vector``). If ``mean`` returns one vector per sample - mean of embedding vectors of tokens.

    - **FasttextEmbedder** (registered as ``fasttext``) reads embedding file in fastText format. If ``mean`` returns one vector per sample - mean of embedding vectors of tokens.

    - **BoWEmbedder** (registered as ``bow``) performs one-hot encoding of tokens using pre-built vocabulary.

    - **TfidfWeightedEmbedder** (registered as ``tfidf_weighted``) accepts embedder, tokenizer (for detokenization, by default, detokenize with joining with space), TFIDF vectorizer or counter vocabulary, optionally accepts tags vocabulary (to assign additional multiplcative weights to particular tags). If ``mean`` returns one vector per sample - mean of embedding vectors of tokens.

    - **ELMoEmbedder** (registered as ``elmo``) converts tokens to pre-trained contextual representations from large-scale bidirectional language models. See examples `here <https://www.tensorflow.org/hub/modules/google/elmo/2>`__.

Vectorizers
-----------

Vectorizer is a component that converts batch of text samples to batch of vectors.

    - **SklearnComponent** (registered as ``sklearn_component``) is a DeepPavlov wrapper for most of sklearn estimators, vectorizers etc. For example, to get TFIDF-vecotrizer one should assign in config ``model_class`` to ``sklearn.feature_extraction.text:TfidfVectorizer``, ``infer_method`` to ``transform``, pass ``load_path``, ``save_path`` and other sklearn model parameters.

    - **HashingTfIdfVectorizer** (registered as ``hashing_tfidf_vectorizer``) implements hashing version of usual TFIDF-vecotrizer. It creates a TFIDF matrix from collection of documents of size ``[n_documents X n_features(hash_size)]``.

