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

from deeppavlov.core.common.file import read_json
from deeppavlov.core.data.utils import download, download_decompress, get_all_elems_from_json
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('--config', '-c', help="path to a pipeline json config", type=str,
                    default=None)
parser.add_argument('-all', action='store_true',
                    help="Download everything. Warning! There should be at least 10 GB space"
                         " available on disk.")
parser.add_argument('-test', action='store_true',
                    help="Turn test mode")


def get_config_downloads(config_path, config_downloads=None):
    config = read_json(config_path)

    if config_downloads is None:
        config_downloads = {}

    if 'metadata' in config and 'download' in config['metadata']:
        for resource in config['metadata']['download']:
            if isinstance(resource, str):
                url = resource
                sub_dir = ''
            elif isinstance(resource, dict):
                url = resource['url']
                sub_dir = resource['subdir'] if 'subdir' in resource else ''

            if url in config_downloads:
                config_downloads[url]['subdir'] = list(set(config_downloads[url]['subdir'] +
                                                           [sub_dir]))
            else:
                config_downloads[url] = {'url': url, 'subdir': [sub_dir]}

    config_references = get_all_elems_from_json(config, 'config_path')
    config_references = [root_path.joinpath(config_ref.split('../', 1)[1]) for config_ref in config_references]

    for config_ref in config_references:
        config_downloads = get_config_downloads(config_ref, config_downloads)

    return config_downloads


def get_configs_downloads(config_path=None, test=None):
    all_downloads = {}

    if test:
        configs_path = root_path / 'tests' / 'deeppavlov' / 'configs'
    else:
        configs_path = root_path / 'deeppavlov' / 'configs'

    if config_path:
        configs = [config_path]
    else:
        configs = list(configs_path.glob('**/*.json'))

    for config_path in configs:
        config_downloads = get_config_downloads(config_path)
        for url in config_downloads:
            if url in all_downloads:
                all_downloads[url]['subdir'] = list(set(all_downloads[url]['subdir'] +
                                                        config_downloads[url]['subdir']))
            else:
                all_downloads[url] = config_downloads[url]

    return all_downloads


def download_resource(resource, download_path):
    url = resource['url']
    sub_dirs = resource['subdir']
    dest_paths = []

    for sub_dir in sub_dirs:
        dest_path = download_path.joinpath(sub_dir)
        dest_paths.append(dest_path)

    if url.endswith(('.tar.gz', '.gz', '.zip')):
        download_path = dest_paths[0].parent
        download_decompress(url, download_path, dest_paths)
    else:
        file_name = url.split('/')[-1]
        dest_files = [dest_path / file_name for dest_path in dest_paths]
        download(dest_files, url)


def download_resources(args):
    download_path = root_path / 'download'

    if args.test:
        download_path = root_path / 'tests' / 'download'
        test = True
    else:
        test = False

    if not args.all and not args.config:
        log.error('You should provide either skill config path or -all flag')
        sys.exit(1)
    elif args.all:
        downloads = get_configs_downloads(test=test)
    else:
        config_path = Path(args.config).resolve()
        downloads = get_configs_downloads(config_path=config_path)

    download_path.mkdir(exist_ok=True)

    for url in downloads:
        resource = downloads[url]
        download_resource(resource, download_path)


def deep_download(args=None):
    args = parser.parse_args(args)
    log.info("Downloading...")
    download_resources(args)
    log.info("\nDownload successful!")


if __name__ == "__main__":
    deep_download()
