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
import shutil
import sys

root_path = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(root_path))

from deeppavlov.core.common.file import read_json
from deeppavlov.core.data.utils import download, download_decompress
from deeppavlov.core.data.urls import REQ_URLS, ALL_URLS, EMBEDDING_URLS, DATA_URLS, BINARY_URLS
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("config_path", help="path to a pipeline json config", type=str)
parser.add_argument('-all', action='store_true',
                    help="Download everything. Warning! There should be at least 10 GB space"
                         " available on disk.")
parser.add_argument('-test', action='store_true',
                    help="Turn test mode")
parser.add_argument('-force', action='store_true',
                    help="Overwrite all existing files")


def get_config_downloads(config_path):
    downloads = {}
    config = read_json(config_path)

    if 'metadata' in config and 'download' in config['metadata']:
        for download in config['metadata']['download']:
            url = download['url']
            downloads[url] = {'compressed': False, 'subdir': []}

            if 'compressed' in download:
                downloads[url]['compressed'] = bool(download['compressed'])

            if 'subdir' in download:
                downloads[url]['subdir'].append(download['subdir'])
            else:
                downloads[url]['subdir'].append('')

    return downloads


def get_all_configs_downloads():
    all_downloads = {}
    configs_path = root_path / 'deeppavlov' / 'configs'
    configs = list(configs_path.glob('**/*.json'))

    for config_path in configs:
        downloads = get_config_downloads(config_path)
        for url in downloads:
            if url in all_downloads:
                all_downloads[url]['compressed'] = downloads[url]['compressed']
                all_downloads[url]['subdir'] = list(set(all_downloads[url]['subdir'] +
                                                        downloads[url]['subdir']))
            else:
                all_downloads[url] = downloads[url]

    return all_downloads


def download_resources(args):
    if args.test:
        download_path = root_path / 'tests' / 'download'
    else:
        download_path = root_path / 'download'

    if not args.all and not args.config_path:
        log.error('You should provide either skill config path or -all flag')
        sys.exit(1)
    elif args.all:
        downloads = get_all_configs_downloads()
    else:
        config_path = Path(args.config_path).resolve()
        downloads = get_config_downloads(config_path)

    download_path.mkdir(exist_ok=True)

    #embeddings_path = download_path.joinpath('embeddings')

    for url in downloads:
        download = downloads[url]
        first_subdir = download['subdir'].pop()
        sub_path = download_path.joinpath(first_subdir)
        sub_path.mkdir(exist_ok=True)

        if download['compressed']:
            dest_path = sub_path.joinpath(url.split('/')[-1].split('.')[0])
            download_decompress(url, dest_path)

            for subdir in download['subdir']:
                sub_path = download_path.joinpath(subdir)
                sub_path.mkdir(exist_ok=True)
                dest_path = sub_path.joinpath(url.split('/')[-1].split('.')[0])

                if dest_path.exists():
                    dest_path.


        else:
            dest_folder = sub_path.joinpath(url.split('/')[-2])
            dest_file = dest_folder.joinpath(url.split('/')[-1])
            download(dest_file, url)

        #dest_path = download_path
        #
        #if url in EMBEDDING_URLS:
        #    embeddings_path.mkdir(exist_ok=True)
        #    dest_path = embeddings_path.joinpath(url.split("/")[-1])
        #    download(dest_path, url)
        #
        #elif url in BINARY_URLS:
        #    dest_folder = download_path.joinpath(url.split("/")[-2])
        #    dest_file = dest_folder.joinpath(url.split("/")[-1])
        #    dest_path.mkdir(exist_ok=True)
        #    download(dest_file, url)
        #
        #elif url in DATA_URLS:
        #    dest_path = download_path.joinpath(url.split("/")[-1].split(".")[0])
        #    download_decompress(url, dest_path)
        #
        #else:
        #    download_decompress(url, dest_path)


def main():
    args = parser.parse_args()
    log.info("Downloading...")
    download_resources(args)
    log.info("\nDownload successful!")


if __name__ == "__main__":
    main()
