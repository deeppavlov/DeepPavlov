from pathlib import Path

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

    dest_file_path = Path(dest_file_path).absolute()
    dest_file_path.parent.mkdir(parents=True, exist_ok=True)

    r = requests.get(source_url, stream=True)
    total_length = int(r.headers.get('content-length', 0))

    with dest_file_path.open('wb') as f:
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
    file_path = Path(file_path)
    if extract_folder is None:
        extract_folder = file_path.parent
    extract_folder = Path(extract_folder)
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
    download_path = Path(download_path)
    if extract_path is None:
        extract_path = download_path
    extract_path = Path(extract_path)
    tar_file_path = download_path / file_name
    print('Extracting {} archive into {}'.format(tar_file_path, extract_path))
    download(tar_file_path, url)
    untar(tar_file_path, extract_path)
    tar_file_path.unlink()


def load_vocab(vocab_path):
    vocab_path = Path(vocab_path)
    with vocab_path.open() as f:
        return f.read().split()


_MARK_DONE = '.done'


def mark_done(path):
    mark = Path(path) / _MARK_DONE
    mark.touch(exist_ok=True)


def is_done(path):
    mark = Path(path) / _MARK_DONE
    return mark.is_file()


def tokenize_reg(s):
    pattern = "[\w]+|[‑–—“”€№…’\"#$%&\'()+,-./:;<>?]"
    return re.findall(re.compile(pattern), s)
