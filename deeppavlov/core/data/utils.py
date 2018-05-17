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
import numpy as np
import re
import zipfile
import shutil

from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)

_MARK_DONE = '.done'

tqdm.monitor_interval = 0


def download(dest_file_path, source_url, force_download=True):
    """Download a file from URL to one or several target locations

    Args:
        dest_file_path: path or list of paths to the file destination files (including file name)
        source_url: the source URL
        force_download: download file if it already exists, or not

    """
    CHUNK = 16 * 1024

    if isinstance(dest_file_path, str):
        dest_file_path = [Path(dest_file_path).absolute()]
    elif isinstance(dest_file_path, Path):
        dest_file_path = [dest_file_path.absolute()]
    elif isinstance(dest_file_path, list):
        dest_file_path = [Path(path) for path in dest_file_path]

    first_dest_path = dest_file_path.pop()

    if force_download or not first_dest_path.exists():
        first_dest_path.parent.mkdir(parents=True, exist_ok=True)

        r = requests.get(source_url, stream=True)
        total_length = int(r.headers.get('content-length', 0))

        with first_dest_path.open('wb') as f:
            log.info('Downloading from {} to {}'.format(source_url, first_dest_path))

            pbar = tqdm(total=total_length, unit='B', unit_scale=True)
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:  # filter out keep-alive new chunks
                    pbar.update(len(chunk))
                    f.write(chunk)
            f.close()
    else:
        log.info('File already exists in {}'.format(first_dest_path))

    while len(dest_file_path) > 0:
        dest_path = dest_file_path.pop()

        if force_download or not dest_path.exists():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(first_dest_path), str(dest_path))
        else:
            log.info('File already exists in {}'.format(dest_path))


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


def download_decompress(url, download_path, extract_paths=None):
    """Download and extract .tar.gz or .gz file to one or several target locations.
    The archive is deleted if extraction was successful.

    Args:
        url: URL for file downloading
        download_path: path to the directory where downloaded file will be stored
        until the end of extraction
        extract_paths: path or list of paths where contents of archive will be extracted
    """
    file_name = url.split('/')[-1]
    download_path = Path(download_path)
    arch_file_path = download_path / file_name
    download(arch_file_path, url)

    if extract_paths is None:
        extract_paths = [download_path]
    elif isinstance(extract_paths, str):
        extract_paths = [Path(extract_paths)]
    elif isinstance(extract_paths, list):
        extract_paths = [Path(path) for path in extract_paths]

    if url.endswith(('.tar.gz', '.gz', '.zip')):
        for extract_path in extract_paths:
            log.info('Extracting {} archive into {}'.format(arch_file_path, extract_path))
            extract_path.mkdir(parents=True, exist_ok=True)

            if url.endswith('.tar.gz'):
                untar(arch_file_path, extract_path)
            elif url.endswith('.gz'):
                ungzip(arch_file_path, extract_path)
            elif url.endswith('.zip'):
                zip_ref = zipfile.ZipFile(arch_file_path, 'r')
                zip_ref.extractall(extract_path)
                zip_ref.close()

        arch_file_path.unlink()
    else:
        log.error('File {} has unsupported format. '
                  'Not extracted, downloaded to {}'.format(file_name, arch_file_path))


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


def zero_pad(batch, dtype=np.float32):
    if len(batch) == 1 and len(batch[0]) == 0:
        return np.array([], dtype=dtype)
    batch_size = len(batch)
    max_len = max(len(utterance) for utterance in batch)
    if isinstance(batch[0][0], (int, np.int)):
        padded_batch = np.zeros([batch_size, max_len], dtype=np.int32)
        for n, utterance in enumerate(batch):
            padded_batch[n, :len(utterance)] = utterance
    else:
        n_features = len(batch[0][0])
        padded_batch = np.zeros([batch_size, max_len, n_features], dtype=dtype)
        for n, utterance in enumerate(batch):
            for k, token_features in enumerate(utterance):
                padded_batch[n, k] = token_features
    return padded_batch


def zero_pad_char(batch, dtype=np.float32):
    if len(batch) == 1 and len(batch[0]) == 0:
        return np.array([], dtype=dtype)
    batch_size = len(batch)
    max_len = max(len(utterance) for utterance in batch)
    max_token_len = max(len(ch) for token in batch for ch in token)
    if isinstance(batch[0][0][0], (int, np.int)):
        padded_batch = np.zeros([batch_size, max_len, max_token_len], dtype=np.int32)
        for n, utterance in enumerate(batch):
            for k, token in enumerate(utterance):
                padded_batch[n, k, :len(token)] = token
    else:
        n_features = len(batch[0][0][0])
        padded_batch = np.zeros([batch_size, max_len, max_token_len, n_features], dtype=dtype)
        for n, utterance in enumerate(batch):
            for k, token in enumerate(utterance):
                for q, char_features in enumerate(token):
                    padded_batch[n, k, q] = char_features
    return padded_batch


def get_all_elems_from_json(search_json, search_key):
    result = []
    if isinstance(search_json, dict):
        for key in search_json:
            if key == search_key:
                result.append(search_json[key])
            else:
                result.extend(get_all_elems_from_json(search_json[key], search_key))
    elif isinstance(search_json, list):
        for item in search_json:
            result.extend(get_all_elems_from_json(item, search_key))

    return result


def check_nested_dict_keys(check_dict: dict, keys: list):
    if isinstance(keys, list) and len(keys) > 0:
        element = check_dict
        for key in keys:
            if isinstance(element, dict) and key in element.keys():
                element = element[key]
            else:
                return False
        return True
    else:
        return False


def jsonify_data(input):
    if isinstance(input, list):
        result = [jsonify_data(item) for item in input]
    elif isinstance(input, tuple):
        result = [jsonify_data(item) for item in input]
    elif isinstance(input, dict):
        result = {}
        for key in input.keys():
            result[key] = jsonify_data(input[key])
    elif isinstance(input, np.ndarray):
        result = input.tolist()
    elif isinstance(input, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        result = int(input)
    elif isinstance(input, (np.float_, np.float16, np.float32, np.float64)):
        result = float(input)
    else:
        result = input
    return result
