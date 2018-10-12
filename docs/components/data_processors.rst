Data processors
===============

Preprocessors
-------------

Preprocessor is a component that processes batch of samples.

Already implemented universal preprocessors of **tokenized texts**:

- **CharSplitter** (registered as ``char_splitter``) splits every token in given batch of tokenized samples to a sequence of characters.

- **Mask** (registered as ``mask``) returns binary mask of corresponding length (padding up to the maximum length per batch.

- **PymorphyRussianLemmatizer** (registered as ``pymorphy_russian_lemmatizer``) performs lemmatization  for Russian language.

- **Sanitizer** (registered as ``sanitizer``) removes all combining characters like diacritical marks from tokens.


Already implemented universal preprocessors of **non-tokenized texts** (each sample is a string):

- **DirtyCommentsPreprocessor** (registered as ``dirty_comments_preprocessor``) preprocesses samples converting samples to lowercase, paraphrasing English combinations with apostrophe ``'``,  transforming more than three the same symbols to two symbols.

- **StrLower** (registered as ``str_lower``) converts samples to lowercase.


Already implemented universal preprocessors of another type of features:

- **OneHotter** (registered as ``one_hotter``) performs one-hotting operation for the batch of samples where each sample is an integer label or a list of integer labels (can be combined in one batch). If ``multi_label`` parameter is set to ``True``, returns one one-dimensional vector per sample with several elements equal to ``1``.

Tokenizers
----------

Gfd  SD GADFGA

Embedders
---------

KLDFSJAG HF

Vectorizers
-----------

SD,L GKHKJDFSGH<
