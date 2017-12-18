from deeppavlov.core.data.utils import is_done, mark_done
import deeppavlov
import os


def untar(file_path, extract_folder=None):
    """Simple tar archive extractor

    Args:
        file_path: path to the tar file to be extracted
        extract_folder: folder to which the files will be extracted

    """
    if extract_folder is None:
        extract_folder = os.path.dirname(file_path)
    tar = tarfile.open(file_path)
    tar.extractall(extract_folder)
    tar.close()


def download_untar(url, download_path, extract_path=None):
    file_name = url.split('/')[-1]
    if extract_path is None:
        extract_path = download_path
    tar_file_path = os.path.join(download_path, file_name)
    print('Extracting {} archive into {}'.format(tar_file_path, extract_path))
    download(tar_file_path, url)
    untar(tar_file_path, extract_path)
    os.remove(tar_file_path)


def load_ner_dstc_model(data_path='model'):
    if not is_done(data_path):
        url = 'http://lnsigo.mipt.ru/export/ner_dstc_model.tar.gz'
        download_untar(url, data_path)
        mark_done(data_path)
