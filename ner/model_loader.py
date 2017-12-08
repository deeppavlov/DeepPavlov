from deeppavlov.data.utils import is_done, mark_done, download
import os

def load_nerpa(dpath):
    if not is_done(dpath):
        url = 'http://lnsigo.mipt.ru/export/ner_dstc_model.tar.gz'
        os.makedirs(dpath, exist_ok=True)
        download(url, dpath, 'dstc_ner_model.tar.gz')
        build_data.untar(dpath, 'dstc_ner_model.tar.gz')
        build_data.mark_done(dpath, version_string=version)