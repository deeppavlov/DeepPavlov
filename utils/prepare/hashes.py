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

import argparse
import gzip
import sys
import tarfile
from hashlib import md5
from pathlib import Path
from typing import List, Dict, Union
from zipfile import ZipFile

from deeppavlov.core.data.utils import file_md5


def tar_md5(fpath: Union[str, Path], chunk_size: int = 2 ** 16) -> Dict[str, str]:
    tar = tarfile.open(fpath)
    res = {}
    while True:
        item: tarfile.TarInfo = tar.next()
        if item is None:
            break
        if not item.isfile():
            continue
        file_hash = md5()
        with tar.extractfile(item) as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                file_hash.update(chunk)
        res[item.name] = file_hash.hexdigest()
    return res


def gzip_md5(fpath: Union[str, Path], chunk_size: int = 2 ** 16) -> str:
    file_hash = md5()
    with gzip.open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def zip_md5(fpath: Union[str, Path], chunk_size: int = 2 ** 16) -> Dict[str, str]:
    res = {}
    with ZipFile(fpath) as zip_f:
        for item in zip_f.infolist():
            if item.is_dir():
                continue
            file_hash = md5()
            with zip_f.open(item) as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    file_hash.update(chunk)
            res[item.filename] = file_hash.hexdigest()
    return res


def compute_hashes(fpath: Union[str, Path]) -> Dict[str, str]:
    p = Path(fpath).expanduser()
    if not p.is_file():
        raise RuntimeError(f'{p} is not a file')

    if '.tar' in {s.lower() for s in p.suffixes}:
        hashes = tar_md5(p)
    elif p.suffix.lower() == '.gz':
        hashes = {p.with_suffix('').name: gzip_md5(p)}
    elif p.suffix.lower() == '.zip':
        hashes = zip_md5(p)
    else:
        hashes = {p.name: file_md5(p)}
    return hashes


def main(args: List[str] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", help="path to a file to compute hash for", type=str)
    parser.add_argument('-o', '--outfile', help='where to write the hashes', default=None, type=str)

    args = parser.parse_args(args)

    p = Path(args.fname).expanduser()
    hashes = compute_hashes(p)

    outfile = args.outfile
    if outfile is None:
        outfile = p.with_suffix(p.suffix + '.md5').open('w', encoding='utf-8')
    elif outfile == '-':
        outfile = sys.stdout
    else:
        outfile = Path(outfile).expanduser().open('w', encoding='utf-8')

    for fname, fhash in hashes.items():
        print(f'{fhash} *{fname}', file=outfile, flush=True)

    if outfile is not sys.stdout:
        outfile.close()


if __name__ == '__main__':
    main()
