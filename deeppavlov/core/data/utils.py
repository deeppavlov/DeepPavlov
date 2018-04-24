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

from pathlib import Path

import requests
from tqdm import tqdm
import tarfile
import gzip
import re
import zipfile

from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)

_MARK_DONE = '.done'

tqdm.monitor_interval = 0


def download(dest_file_path, source_url, force_download=True):
    """Download a file from URL

    Args:
        dest_file_path: path to the file destination file (including file name)
        source_url: the source URL
        force_download: download file if it already exists, or not

    """
    CHUNK = 16 * 1024
    dest_file_path = Path(dest_file_path).absolute()

    if force_download or not dest_file_path.exists():
        dest_file_path.parent.mkdir(parents=True, exist_ok=True)

        r = requests.get(source_url, stream=True)
        total_length = int(r.headers.get('content-length', 0))

        with dest_file_path.open('wb') as f:
            log.info('Downloading from {} to {}'.format(source_url, dest_file_path))

            pbar = tqdm(total=total_length, unit='B', unit_scale=True)
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:  # filter out keep-alive new chunks
                    pbar.update(len(chunk))
                    f.write(chunk)
            f.close()
    else:
        log.info('File already exists in {}'.format(dest_file_path))


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


def ungzip(file_path, extract_folder=None):
    """Simple .gz archive extractor

        Args:
            file_path: path to the gzip file to be extracted
            extract_folder: folder to which the files will be extracted

        """
    CHUNK = 16 * 1024
    file_path = Path(file_path)
    extract_path = file_path.with_suffix('')
    if extract_folder is not None:
        extract_path = Path(extract_folder) / extract_path.name

    with gzip.open(file_path, 'rb') as fin, extract_path.open('wb') as fout:
        while True:
            block = fin.read(CHUNK)
            if not block:
                break
            fout.write(block)


def download_decompress(url, download_path, extract_path=None):
    """Download and extract .tar.gz or .gz file. The archive is deleted after extraction.

    Args:
        url: URL for file downloading
        download_path: path to the directory where downloaded file will be stored
        until the end of extraction
        extract_path: path where contents of archive will be extracted
    """
    file_name = url.split('/')[-1]
    download_path = Path(download_path)
    if extract_path is None:
        extract_path = download_path
    extract_path = Path(extract_path)
    arch_file_path = download_path / file_name
    log.info('Extracting {} archive into {}'.format(arch_file_path, extract_path))
    download(arch_file_path, url)
    if url.endswith('.tar.gz'):
        untar(arch_file_path, extract_path)
    elif url.endswith('.gz'):
        ungzip(arch_file_path, extract_path)
    elif url.endswith('.zip'):
        zip_ref = zipfile.ZipFile(arch_file_path, 'r')
        zip_ref.extractall(extract_path)
        zip_ref.close()
    arch_file_path.unlink()


def load_vocab(vocab_path):
    vocab_path = Path(vocab_path)
    with vocab_path.open() as f:
        return f.read().split()


def mark_done(path):
    mark = Path(path) / _MARK_DONE
    mark.touch(exist_ok=True)


def is_done(path):
    mark = Path(path) / _MARK_DONE
    return mark.is_file()


def tokenize_reg(s):
    pattern = "[\w]+|[‑–—“”€№…’\"#$%&\'()+,-./:;<>?]"
    return re.findall(re.compile(pattern), s)
