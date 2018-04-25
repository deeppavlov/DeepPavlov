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


def get_config_downloads(config_path):
    config_downloads = {}
    config = read_json(config_path)

    if 'metadata' in config and 'download' in config['metadata']:
        for resource in config['metadata']['download']:
            resource_info = {}

            if isinstance(resource, str):
                url = resource
                resource_info['url'] = url
                resource_info['subdir'] = ['']

            elif isinstance(resource_info, dict):
                url = resource['url']
                resource_info['url'] = url

                if 'subdir' in resource:
                    resource_info['subdir'] = [resource['subdir']]
                else:
                    resource_info['subdir'] = ['']

            config_downloads[url] = resource_info

    return config_downloads


def get_configs_downloads(config_path=None):
    all_downloads = {}

    if config_path:
        configs = [config_path]
    else:
        configs_path = root_path / 'deeppavlov' / 'configs'
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
    if args.test:
        download_path = root_path / 'tests' / 'download'
    else:
        download_path = root_path / 'download'

    if not args.all and not args.config:
        log.error('You should provide either skill config path or -all flag')
        sys.exit(1)
    elif args.all:
        downloads = get_configs_downloads()
    else:
        config_path = Path(args.config).resolve()
        downloads = get_configs_downloads(config_path)

    download_path.mkdir(exist_ok=True)

    for url in downloads:
        import pprint
        pprint.pprint(downloads)

        resource = downloads[url]
        download_resource(resource, download_path)


def main():
    args = parser.parse_args()
    log.info("Downloading...")
    download_resources(args)
    log.info("\nDownload successful!")


if __name__ == "__main__":
    main()
