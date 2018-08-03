"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

REQ_URLS = {
    'http://files.deeppavlov.ai/deeppavlov_data/gobot_dstc2_v3.tar.gz',
    'http://files.deeppavlov.ai/deeppavlov_data/gobot_dstc2_best_v1.tar.gz',
    'http://files.deeppavlov.ai/deeppavlov_data/embeddings/wiki.en.bin',
    'http://files.deeppavlov.ai/deeppavlov_data/intents.tar.gz',
    'http://files.deeppavlov.ai/deeppavlov_data/slotfill_dstc2.tar.gz',
    'http://files.deeppavlov.ai/deeppavlov_data/ner_conll2003_v2.tar.gz',
    'http://files.deeppavlov.ai/deeppavlov_data/error_model.tar.gz',
    'http://files.deeppavlov.ai/datasets/insuranceQA-master.zip',
    'http://files.deeppavlov.ai/deeppavlov_data/insurance_ranking.tar.gz',
    'http://files.deeppavlov.ai/embeddings/glove.6B.100d.txt',
    'http://files.deeppavlov.ai/embeddings/lenta_lower_100.bin',
    'http://files.deeppavlov.ai/deeppavlov_data/vocabs.tar.gz',
    'http://files.deeppavlov.ai/deeppavlov_data/slots.tar.gz',
    'http://files.deeppavlov.ai/deeppavlov_data/embeddings/dstc2_fastText_model.bin',
    'http://files.deeppavlov.ai/datasets/dstc2.tar.gz',
    'http://files.deeppavlov.ai/deeppavlov_data/squad_model_1.1.tar.gz',
    'http://files.deeppavlov.ai/deeppavlov_data/squad_model_ru.tar.gz',
    'http://files.deeppavlov.ai/deeppavlov_data/seq2seq_go_bot.tar.gz',
    'http://files.deeppavlov.ai/deeppavlov_data/odqa.tar.gz',
    'http://files.deeppavlov.ai/deeppavlov_data/ner_ontonotes_v2.tar.gz',
    'http://files.deeppavlov.ai/deeppavlov_data/senna.tar.gz'
    'http://files.deeppavlov.ai/deeppavlov_data/ner_rus.tar.gz'
    'http://files.deeppavlov.ai/deeppavlov_data/morpho_tagger.tar.gz'
}

OPT_URLS = {
    'http://files.deeppavlov.ai/embeddings/wiki-news-300d-1M.vec',
    'http://files.deeppavlov.ai/embeddings/wiki-news-300d-1M-char.vec',
    'http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize-char.vec',
    'http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize/ft_native_300_ru_wiki_lenta_nltk_word_tokenize.vec',
    'http://files.deeppavlov.ai/deeppavlov_data/odqa.tar.gz'
}

ALL_URLS = REQ_URLS.union(OPT_URLS)

EMBEDDING_URLS = {
    'http://files.deeppavlov.ai/deeppavlov_data/embeddings/wiki.en.bin',
    'http://files.deeppavlov.ai/deeppavlov_data/embeddings/dstc2_fastText_model.bin',
    'http://files.deeppavlov.ai/embeddings/wiki-news-300d-1M.vec',
    'http://files.deeppavlov.ai/embeddings/wiki-news-300d-1M-char.vec',
    'http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize-char.vec',
    'http://files.deeppavlov.ai/embeddings/glove.6B.100d.txt'
}

# put there urls that do not require decompression:
BINARY_URLS = {
   # 'http://files.deeppavlov.ai/deeppavlov_data/odqa/wiki_tfidf_matrix.npz'
}

DATA_URLS = {
    'http://files.deeppavlov.ai/datasets/dstc2.tar.gz',
    'http://files.deeppavlov.ai/datasets/insuranceQA-master.zip'
}
