from deeppavlov.data.utils import is_done, mark_done, download_untar
import deeppavlov
import os


def load_ner_dstc_model(data_path='model'):
    if not is_done(data_path):
        url = 'http://lnsigo.mipt.ru/export/ner_dstc_model.tar.gz'
        download_untar(url, data_path)
        mark_done(data_path)
