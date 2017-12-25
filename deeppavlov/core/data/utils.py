import os
import requests
from tqdm import tqdm
import tarfile
import re


def download(dest_file_path, source_url):
    """Download a file from URL

    Args:
        dest_file_path: path to the file destination file (including file name)
        source_url: the source URL

    """
    datapath = os.path.dirname(dest_file_path)
    os.makedirs(datapath, mode=0o755, exist_ok=True)

    dest_file_path = os.path.abspath(dest_file_path)

    r = requests.get(source_url, stream=True)
    total_length = int(r.headers.get('content-length', 0))

    with open(dest_file_path, 'wb') as f:
        print('Downloading from {} to {}'.format(source_url, dest_file_path))

        pbar = tqdm(total=total_length, unit='B', unit_scale=True)
        for chunk in r.iter_content(chunk_size=32 * 1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)


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
    """Download and extract tar.gz file. The archive is deleted after extraction.

    Args:
        url: URL for file downloading
        download_path: path to the directory where downloaded file will be stored until the end of extraction
        extract_path: path where contents of tar file will be extracted
    """
    file_name = url.split('/')[-1]
    if extract_path is None:
        extract_path = download_path
    tar_file_path = os.path.join(download_path, file_name)
    print('Extracting {} archive into {}'.format(tar_file_path, extract_path))
    download(tar_file_path, url)
    untar(tar_file_path, extract_path)
    os.remove(tar_file_path)


def load_vocab(vocab_path):
    with open(vocab_path) as f:
        return f.read().split()


_MARK_DONE = '.done'


def mark_done(path):
    fname = os.path.join(path, _MARK_DONE)
    with open(fname, 'a'):
        os.utime(fname, None)


def is_done(path):
    return os.path.isfile(os.path.join(path, _MARK_DONE))


def tokenize_reg(s):
    pattern = "[\w]+|[‑–—“”€№…’\"#$%&\'()+,-./:;<>?]"
    return re.findall(re.compile(pattern), s)
