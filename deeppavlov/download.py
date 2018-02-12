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

import argparse
from pathlib import Path
import sys

root_path = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(root_path))

from deeppavlov.core.data.utils import download, download_decompress
from deeppavlov.core.data.urls import REQ_URLS, ALL_URLS, EMBEDDING_URLS, DATA_URLS
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('-all', action='store_true',
                    help="Download everything. Warning! There should be at least 10 GB space"
                         " available on disk.")


def download_resources(args):
    if args.all:
        urls = ALL_URLS
    else:
        urls = REQ_URLS

    for url in urls:
        download_path = root_path / 'download'
        download_path.mkdir(exist_ok=True)
        dest_path = download_path

        embeddings_path = download_path.joinpath('embeddings')

        if url in EMBEDDING_URLS:
            embeddings_path.mkdir(exist_ok=True)
            dest_path = embeddings_path.joinpath(url.split("/")[-1])
            download(dest_path, url)

        elif url in DATA_URLS:
            dest_path = download_path.joinpath(url.split("/")[-1].split(".")[0])
            download_decompress(url, dest_path)

        else:
            download_decompress(url, dest_path)


def main():
    args = parser.parse_args()
    log.info("Downloading...")
    download_resources(args)
    log.info("\nDownload successful!")


if __name__ == "__main__":
    main()
