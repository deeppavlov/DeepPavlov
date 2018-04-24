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
parser.add_argument('-force', action='store_true',
                    help="Overwrite existing downloaded files")


def get_config_downloads(config_path):
    config_downloads = {}
    config = read_json(config_path)

    if 'metadata' in config and 'download' in config['metadata']:
        for resource in config['metadata']['download']:
            config_downloads = {
                'url': resource['url'],
                'compressed': False,
                'subdir': []
            }

            if 'compressed' in resource:
                config_downloads['compressed'] = bool(resource['compressed'])

            if 'subdir' in resource:
                config_downloads['subdir'].append(resource['subdir'])
            else:
                config_downloads['subdir'].append('')

    return config_downloads


def get_all_configs_downloads():
    all_downloads = {}
    configs_path = root_path / 'deeppavlov' / 'configs'
    configs = list(configs_path.glob('**/*.json'))

    for config_path in configs:
        downloads = get_config_downloads(config_path)
        for url in downloads:
            if url in all_downloads:
                all_downloads[url]['compressed'] = downloads['compressed']
                all_downloads[url]['subdir'] = list(set(all_downloads[url]['subdir'] +
                                                        downloads['subdir']))
            else:
                all_downloads[url] = downloads

    return all_downloads.items()


def get_destination_paths(url, download_path, sub_dir, compressed=False):
    sub_path = download_path.joinpath(sub_dir)
    sub_path.mkdir(exist_ok=True)

    if compressed:
        dest_path = sub_path.joinpath(url.split('/')[-1].split('.')[0])
        dest_paths = (dest_path, )
    else:
        dest_path = sub_path.joinpath(url.split('/')[-2])
        dest_file = sub_path.joinpath(url.split('/')[-1])
        dest_paths = (dest_path, dest_file)

    return dest_paths


def download_resource(resource, download_path, force_donwload=False):
    url = resource['url']
    compressed = resource['compressed']
    first_sub_dir = resource['subdir'].pop()

    first_dest_paths = get_destination_paths(url, download_path, first_sub_dir, compressed)
    first_dest_dir = first_dest_paths[0]
    if len(first_dest_paths) > 1:
        first_dest_file = first_dest_paths[1]

    if force_donwload or not first_dest_dir.exists():
        if first_dest_dir.exists():
            shutil.rmtree(str(first_dest_dir), ignore_errors=True)

        if compressed:
            download_decompress(url, first_dest_dir)
        else:
            first_dest_dir.mkdir()
            download(first_dest_file, url)

    for sub_dir in resource['subdir']:
        dest_paths = get_destination_paths(url, download_path, sub_dir, compressed)
        dest_dir = dest_paths[0]

        if force_donwload or not dest_dir.exists():
            if dest_dir.exists():
                shutil.rmtree(str(dest_dir), ignore_errors=True)
            shutil.copy(str(first_dest_dir), str(dest_dir))


def download_resources(args):
    if args.test:
        download_path = root_path / 'tests' / 'download'
    else:
        download_path = root_path / 'download'

    if not args.all and not args.config:
        log.error('You should provide either skill config path or -all flag')
        sys.exit(1)
    elif args.all:
        downloads = get_all_configs_downloads()
    else:
        config_path = Path(args.config).resolve()
        downloads = [get_config_downloads(config_path)]

    download_path.mkdir(exist_ok=True)

    force_download = args.force
    for resource in downloads:
        download_resource(resource, download_path, force_donwload=force_download)


def main():
    args = parser.parse_args()
    log.info("Downloading...")
    download_resources(args)
    log.info("\nDownload successful!")


if __name__ == "__main__":
    main()
