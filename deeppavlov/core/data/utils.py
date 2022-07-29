# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import gzip
import os
import secrets
import shutil
import tarfile
import zipfile
from hashlib import md5
from itertools import chain
from logging import getLogger
from pathlib import Path
from typing import Any, Generator, Iterable, List, Mapping, Optional, Sequence, Sized, Union, Collection
from urllib.parse import urlencode, parse_qs, urlsplit, urlunsplit, urlparse

import numpy as np
import requests
from tqdm import tqdm

log = getLogger(__name__)

_MARK_DONE = '.done'

tqdm.monitor_interval = 0


def get_download_token() -> str:
    """Return a download token from ~/.deeppavlov/token file.

    If token file does not exists, creates the file and writes to it a random URL-safe text string
    containing 32 random bytes.

    Returns:
        32 byte URL-safe text string from ~/.deeppavlov/token.

    """
    token_file = Path.home() / '.deeppavlov' / 'token'
    if not token_file.exists():
        if token_file.parent.is_file():
            token_file.parent.unlink()
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(secrets.token_urlsafe(32), encoding='utf8')

    return token_file.read_text(encoding='utf8').strip()


def s3_download(url: str, destination: str) -> None:
    """Download a file from an Amazon S3 path `s3://<bucket_name>/<key>`

    Requires the boto3 library to be installed and AWS credentials being set
    via environment variables or a credentials file

    Args:
        url: The source URL.
        destination: Path to the file destination (including file name).
    """
    import boto3

    s3 = boto3.resource('s3', endpoint_url=os.environ.get('AWS_ENDPOINT_URL'))

    bucket, key = url[5:].split('/', maxsplit=1)
    file_object = s3.Object(bucket, key)
    file_size = file_object.content_length
    with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
        file_object.download_file(destination, Callback=pbar.update)


def simple_download(url: str, destination: Union[Path, str], headers: Optional[dict] = None) -> None:
    """Download a file from URL to target location.

    Displays a progress bar to the terminal during the download process.

    Args:
        url: The source URL.
        destination: Path to the file destination (including file name).
        headers: Headers for file server.

    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    log.info('Downloading from {} to {}'.format(url, destination))

    if url.startswith('s3://'):
        return s3_download(url, str(destination))

    chunk_size = 32 * 1024
    temporary = destination.with_suffix(destination.suffix + '.part')

    r = requests.get(url, stream=True, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f'Got status code {r.status_code} when trying to download {url}')
    total_length = int(r.headers.get('content-length', 0))

    if temporary.exists() and temporary.stat().st_size > total_length:
        temporary.write_bytes(b'')  # clearing temporary file when total_length is inconsistent

    with temporary.open('ab') as f:
        downloaded = f.tell()
        if downloaded != 0:
            log.warning(f'Found a partial download {temporary}')
        with tqdm(initial=downloaded, total=total_length, unit='B', unit_scale=True) as pbar:
            while True:
                if downloaded != 0:
                    log.warning(f'Download stopped abruptly, trying to resume from {downloaded} '
                                f'to reach {total_length}')
                    headers['Range'] = f'bytes={downloaded}-'
                    r = requests.get(url, headers=headers, stream=True)
                    if 'content-length' not in r.headers or \
                            total_length - downloaded != int(r.headers['content-length']):
                        raise RuntimeError('It looks like the server does not support resuming downloads.')

                try:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:  # filter out keep-alive new chunks
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
                            f.write(chunk)
                except requests.exceptions.ChunkedEncodingError:
                    if downloaded == 0:
                        r = requests.get(url, stream=True, headers=headers)

                if downloaded >= total_length:
                    # Note that total_length is 0 if the server didn't return the content length,
                    # in this case we perform just one iteration and assume that we are done.
                    break

    temporary.rename(destination)


def download(dest_file_path: [List[Union[str, Path]]], source_url: str, force_download: bool = True,
             headers: Optional[dict] = None) -> None:
    """Download a file from URL to one or several target locations.

    Args:
        dest_file_path: Path or list of paths to the file destination (including file name).
        source_url: The source URL.
        force_download: Download file if it already exists, or not.
        headers: Headers for file server.

    """

    if isinstance(dest_file_path, list):
        dest_file_paths = [Path(path) for path in dest_file_path]
    else:
        dest_file_paths = [Path(dest_file_path).absolute()]

    if not force_download:
        to_check = list(dest_file_paths)
        dest_file_paths = []
        for p in to_check:
            if p.exists():
                log.info(f'File already exists in {p}')
            else:
                dest_file_paths.append(p)

    if dest_file_paths:
        cache_dir = os.getenv('DP_CACHE_DIR')
        cached_exists = False
        if cache_dir:
            first_dest_path = Path(cache_dir) / md5(source_url.encode('utf8')).hexdigest()[:15]
            cached_exists = first_dest_path.exists()
        else:
            first_dest_path = dest_file_paths.pop()

        if not cached_exists:
            first_dest_path.parent.mkdir(parents=True, exist_ok=True)

            simple_download(source_url, first_dest_path, headers)
        else:
            log.info(f'Found cached {source_url} in {first_dest_path}')

        for dest_path in dest_file_paths:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(first_dest_path), str(dest_path))


def untar(file_path: Union[Path, str], extract_folder: Optional[Union[Path, str]] = None) -> None:
    """Simple tar archive extractor.

    Args:
        file_path: Path to the tar file to be extracted.
        extract_folder: Folder to which the files will be extracted.

    """
    file_path = Path(file_path)
    if extract_folder is None:
        extract_folder = file_path.parent
    extract_folder = Path(extract_folder)
    tar = tarfile.open(file_path)
    tar.extractall(extract_folder)
    tar.close()


def ungzip(file_path: Union[Path, str], extract_path: Optional[Union[Path, str]] = None) -> None:
    """Simple .gz archive extractor.

    Args:
        file_path: Path to the gzip file to be extracted.
        extract_path: Path where the file will be extracted.

    """
    chunk_size = 16 * 1024
    file_path = Path(file_path)
    if extract_path is None:
        extract_path = file_path.with_suffix('')
    extract_path = Path(extract_path)

    with gzip.open(file_path, 'rb') as fin, extract_path.open('wb') as fout:
        while True:
            block = fin.read(chunk_size)
            if not block:
                break
            fout.write(block)


def download_decompress(url: str,
                        download_path: Union[Path, str],
                        extract_paths: Optional[Union[List[Union[Path, str]], Path, str]] = None,
                        headers: Optional[dict] = None) -> None:
    """Download and extract .tar.gz or .gz file to one or several target locations.

    The archive is deleted if extraction was successful.

    Args:
        url: URL for file downloading.
        download_path: Path to the directory where downloaded file will be stored until the end of extraction.
        extract_paths: Path or list of paths where contents of archive will be extracted.
        headers: Headers for file server.

    """
    file_name = Path(urlparse(url).path).name
    download_path = Path(download_path)

    if extract_paths is None:
        extract_paths = [download_path]
    elif isinstance(extract_paths, list):
        extract_paths = [Path(path) for path in extract_paths]
    else:
        extract_paths = [Path(extract_paths)]

    cache_dir = os.getenv('DP_CACHE_DIR')
    extracted = False
    if cache_dir:
        cache_dir = Path(cache_dir)
        url_hash = md5(url.encode('utf8')).hexdigest()[:15]
        arch_file_path = cache_dir / url_hash
        extracted_path = cache_dir / (url_hash + '_extracted')
        extracted = extracted_path.exists()
        if not extracted and not arch_file_path.exists():
            simple_download(url, arch_file_path, headers)
        else:
            if extracted:
                log.info(f'Found cached and extracted {url} in {extracted_path}')
            else:
                log.info(f'Found cached {url} in {arch_file_path}')
    else:
        arch_file_path = download_path / file_name
        simple_download(url, arch_file_path, headers)
        extracted_path = extract_paths.pop()

    if not extracted:
        log.info('Extracting {} archive into {}'.format(arch_file_path, extracted_path))
        extracted_path.mkdir(parents=True, exist_ok=True)

        if file_name.endswith('.tar.gz'):
            untar(arch_file_path, extracted_path)
        elif file_name.endswith('.gz'):
            ungzip(arch_file_path, extracted_path / Path(file_name).with_suffix('').name)
        elif file_name.endswith('.zip'):
            with zipfile.ZipFile(arch_file_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_path)
        else:
            raise RuntimeError(f'Trying to extract an unknown type of archive {file_name}')

        if not cache_dir:
            arch_file_path.unlink()

    for extract_path in extract_paths:
        for src in extracted_path.iterdir():
            dest = extract_path / src.name
            if src.is_dir():
                _copytree(src, dest)
            else:
                extract_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src), str(dest))


def _copytree(src: Path, dest: Path) -> None:
    """Recursively copies directory.

    Destination directory could exist (unlike if we used shutil.copytree).

    Args:
        src: Path to copied directory.
        dest: Path to destination directory.

    """
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        f_dest = dest / f.name
        if f.is_dir():
            _copytree(f, f_dest)
        else:
            shutil.copy(str(f), str(f_dest))


def file_md5(fpath: Union[str, Path], chunk_size: int = 2 ** 16) -> Optional[str]:
    """Return md5 hash value for file contents.

    Args:
        fpath: Path to file.
        chunk_size: md5 object updated by ``chunk_size`` bytes from file.

    Returns:
        None if ``fpath`` does not point to a file, else returns md5 hash value as string.

    """
    fpath = Path(fpath)
    if not fpath.is_file():
        return None
    file_hash = md5()
    with fpath.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def mark_done(path: Union[Path, str]) -> None:
    """Create ``.done`` empty file in the directory.

    Args:
        path: Path to directory.

    Raises:
        NotADirectoryError: If ``path`` does not point to a directory.

    """
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: '{path}'")
    mark = path / _MARK_DONE
    mark.touch(exist_ok=True)


def is_done(path: Union[Path, str]) -> bool:
    """Check if ``.done`` file exists in directory.

    Args:
        path: Path to directory.

    Returns:
        True if directory contains ``.done`` file, False otherwise.

    """
    mark = Path(path) / _MARK_DONE
    return mark.is_file()


def _get_all_dimensions(batch: Sequence, level: int = 0, res: Optional[List[List[int]]] = None) -> List[List[int]]:
    """Return all presented element sizes of each dimension.

    Args:
        batch: Data array.
        level: Recursion level.
        res: List containing element sizes of each dimension.

    Return:
        List, i-th element of which is list containing all presented sized of batch's i-th dimension.

    Examples:
        >>> x = [[[1], [2, 3]], [[4], [5, 6, 7], [8, 9]]]
        >>> _get_all_dimensions(x)
        [[2], [2, 3], [1, 2, 1, 3, 2]]

    """
    if not level:
        res = [[len(batch)]]
    if len(batch) and isinstance(batch[0], Sized) and not isinstance(batch[0], str):
        level += 1
        if len(res) <= level:
            res.append([])
        for item in batch:
            res[level].append(len(item))
            _get_all_dimensions(item, level, res)
    return res


def get_dimensions(batch: Sequence) -> List[int]:
    """Return maximal size of each batch dimension."""
    return list(map(max, _get_all_dimensions(batch)))


def zero_pad(batch: Sequence,
             zp_batch: Optional[np.ndarray] = None,
             dtype: type = np.float32,
             padding: Union[int, float] = 0) -> np.ndarray:
    """Fills the end of each array item to make its length maximal along each dimension.

    Args:
        batch: Initial array.
        zp_batch: Padded array.
        dtype = Type of padded array.
        padding = Number to will initial array with.

    Returns:
        Padded array.

    Examples:
        >>> x = np.array([[1, 2, 3], [4], [5, 6]])
        >>> zero_pad(x)
        array([[1., 2., 3.],
               [4., 0., 0.],
               [5., 6., 0.]], dtype=float32)

    """
    if zp_batch is None:
        dims = get_dimensions(batch)
        zp_batch = np.ones(dims, dtype=dtype) * padding
    if zp_batch.ndim == 1:
        zp_batch[:len(batch)] = batch
    else:
        for b, zp in zip(batch, zp_batch):
            zero_pad(b, zp)
    return zp_batch


def is_str_batch(batch: Iterable) -> bool:
    """Checks if iterable argument contains string at any nesting level."""
    while True:
        if isinstance(batch, Iterable):
            if isinstance(batch, str):
                return True
            elif isinstance(batch, np.ndarray):
                return batch.dtype.kind == 'U'
            else:
                if len(batch) > 0:
                    batch = batch[0]
                else:
                    return True
        else:
            return False


def flatten_str_batch(batch: Union[str, Iterable]) -> Union[list, chain]:
    """Joins all strings from nested lists to one ``itertools.chain``.

    Args:
        batch: List with nested lists to flatten.

    Returns:
        Generator of flat List[str]. For str ``batch`` returns [``batch``].

    Examples:
        >>> [string for string in flatten_str_batch(['a', ['b'], [['c', 'd']]])]
        ['a', 'b', 'c', 'd']

    """
    if isinstance(batch, str):
        return [batch]
    else:
        return chain(*[flatten_str_batch(sample) for sample in batch])


def zero_pad_truncate(batch: Sequence[Sequence[Union[int, float, np.integer, np.floating,
                                                     Sequence[Union[int, float, np.integer, np.floating]]]]],
                      max_len: int, pad: str = 'post', trunc: str = 'post',
                      dtype: Optional[Union[type, str]] = None) -> np.ndarray:
    """

    Args:
        batch: assumes a batch of lists of word indexes or their vector representations
        max_len: resulting length of every batch item
        pad: how to pad shorter batch items: can be ``'post'`` or ``'pre'``
        trunc: how to truncate a batch item: can be ``'post'`` or ``'pre'``
        dtype: overrides dtype for the resulting ``ndarray`` if specified,
         otherwise ``np.int32`` is used for 2-d arrays and ``np.float32`` — for 3-d arrays

    Returns:
        a 2-d array of size ``(len(batch), max_len)`` or a 3-d array of size ``(len(batch), max_len, len(batch[0][0]))``
    """
    if isinstance(batch[0][0], Collection):  # ndarray behaves like a Sequence without actually being one
        size = (len(batch), max_len, len(batch[0][0]))
        dtype = dtype or np.float32
    else:
        size = (len(batch), max_len)
        dtype = dtype or np.int32

    padded_batch = np.zeros(size, dtype=dtype)
    for i, batch_item in enumerate(batch):
        if len(batch_item) > max_len:  # trunc
            padded_batch[i] = batch_item[slice(max_len) if trunc == 'post' else slice(-max_len, None)]
        else:  # pad
            padded_batch[i, slice(len(batch_item)) if pad == 'post' else slice(-len(batch_item), None)] = batch_item

    return np.asarray(padded_batch)


def get_all_elems_from_json(search_json: dict, search_key: str) -> list:
    """Returns values by key in all nested dicts.

    Args:
        search_json: Dictionary in which one needs to find all values by specific key.
        search_key: Key for search.

    Returns:
        List of values stored in nested structures by ``search_key``.

    Examples:
        >>> get_all_elems_from_json({'a':{'b': [1,2,3]}, 'b':42}, 'b')
        [[1, 2, 3], 42]

    """
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


def check_nested_dict_keys(check_dict: dict, keys: list) -> bool:
    """Checks if dictionary contains nested keys from keys list.

    Args:
        check_dict: Dictionary to check.
        keys: Keys list. i-th nested dict of ``check_dict`` should contain dict containing (i+1)-th key
        from the ``keys`` list by i-th key.

    Returns:
        True if dictionary contains nested keys from keys list, False otherwise.

    Examples:
        >>> check_nested_dict_keys({'x': {'y': {'z': 42}}}, ['x', 'y', 'z'])
        True
        >>> check_nested_dict_keys({'x': {'y': {'z': 42}}}, ['x', 'z', 'y'])
        False
        >>> check_nested_dict_keys({'x': {'y': 1, 'z': 42}}, ['x', 'y', 'z'])
        False

    """
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


def jsonify_data(data: Any) -> Any:
    """Replaces JSON-non-serializable objects with JSON-serializable.

    Function replaces numpy arrays and numbers with python lists and numbers, tuples is replaces with lists. All other
    object types remain the same.

    Args:
        data: Object to make JSON-serializable.

    Returns:
        Modified input data.

    """
    if isinstance(data, (list, tuple)):
        result = [jsonify_data(item) for item in data]
    elif isinstance(data, dict):
        result = {}
        for key in data.keys():
            result[key] = jsonify_data(data[key])
    elif isinstance(data, np.ndarray):
        result = data.tolist()
    elif isinstance(data, np.integer):
        result = int(data)
    elif isinstance(data, np.floating):
        result = float(data)
    elif callable(getattr(data, "to_serializable_dict", None)):
        result = data.to_serializable_dict()
    else:
        result = data
    return result


def chunk_generator(items_list: list, chunk_size: int) -> Generator[list, None, None]:
    """Yields consecutive slices of list.

    Args:
        items_list: List to slice.
        chunk_size: Length of slice.

    Yields:
        list: ``items_list`` consecutive slices.

    """
    for i in range(0, len(items_list), chunk_size):
        yield items_list[i:i + chunk_size]


def update_dict_recursive(editable_dict: dict, editing_dict: Mapping) -> None:
    """Updates dict recursively.

    You need to use this function to update dictionary if depth of editing_dict is more then 1.

    Args:
        editable_dict: Dictionary to edit.
        editing_dict: Dictionary containing edits.

    """
    for k, v in editing_dict.items():
        if isinstance(v, collections.Mapping):
            update_dict_recursive(editable_dict.get(k, {}), v)
        else:
            editable_dict[k] = v


def path_set_md5(url: str) -> str:
    """Given a file URL, return a md5 query of the file.

    Args:
        url: A given URL.

    Returns:
        URL of the md5 file.

    """
    scheme, netloc, path, query_string, fragment = urlsplit(url)
    path += '.md5'

    return urlunsplit((scheme, netloc, path, query_string, fragment))


def set_query_parameter(url: str, param_name: str, param_value: str) -> str:
    """Given a URL, set or replace a query parameter and return the modified URL.

    Args:
        url: A given  URL.
        param_name: The parameter name to add.
        param_value: The parameter value.

    Returns:
        URL with the added parameter.

    """
    scheme, netloc, path, query_string, fragment = urlsplit(url)
    query_params = parse_qs(query_string)

    query_params[param_name] = [param_value]
    new_query_string = urlencode(query_params, doseq=True)

    return urlunsplit((scheme, netloc, path, new_query_string, fragment))
