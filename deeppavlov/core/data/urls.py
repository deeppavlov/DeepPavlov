REQ_URLS = {
    'http://lnsigo.mipt.ru/export/deeppavlov_data/go_bot_rnn.tar.gz',
    'http://lnsigo.mipt.ru/export/deeppavlov_data/intents.tar.gz',
    'http://lnsigo.mipt.ru/export/deeppavlov_data/ner.tar.gz',
    'http://lnsigo.mipt.ru/export/deeppavlov_data/error_model.tar.gz',
    'http://lnsigo.mipt.ru/export/deeppavlov_data/vocabs.tar.gz',
    'http://lnsigo.mipt.ru/export/deeppavlov_data/slots.tar.gz',
    'http://lnsigo.mipt.ru/export/deeppavlov_data/embeddings/dstc2_fasttext_model_100.bin',
    'http://lnsigo.mipt.ru/export/datasets/dstc2.tar.gz'
}

OPT_URLS = {
    'http://lnsigo.mipt.ru/export/deeppavlov_data/embeddings/wiki.en.bin'
}

ALL_URLS = REQ_URLS.union(OPT_URLS)

EMBEDDING_URLS = {
    'http://lnsigo.mipt.ru/export/deeppavlov_data/embeddings/wiki.en.bin',
    'http://lnsigo.mipt.ru/export/deeppavlov_data/embeddings/dstc2_fasttext_model_100.bin'
}

DATA_URLS = {
    'http://lnsigo.mipt.ru/export/datasets/dstc2.tar.gz'
}
