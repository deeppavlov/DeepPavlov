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

import secrets
import shutil
import sys
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Union, Optional, Dict, Iterable, Set, Tuple, List
from urllib.parse import urlparse
import requests
from filelock import FileLock

import deeppavlov
from deeppavlov.core.commands.utils import expand_path, parse_config
from deeppavlov.core.data.utils import download, download_decompress, get_all_elems_from_json, file_md5, \
    set_query_parameter, path_set_md5, get_download_token

log = getLogger(__name__)

parser = ArgumentParser()

parser.add_argument('--config', '-c', help="path to a pipeline json config", type=str,
                    default=None)
parser.add_argument('-all', action='store_true',
                    help="Download everything. Warning! There should be at least 10 GB space"
                         " available on disk.")


def get_config_downloads(config: Union[str, Path, dict]) -> Set[Tuple[str, Path]]:
    config = parse_config(config)

    downloads = set()
    if 'metadata' in config and 'download' in config['metadata']:
        for resource in config['metadata']['download']:
            if isinstance(resource, str):
                resource = {
                    'url': resource
                }

            url = resource['url']
            dest = expand_path(resource.get('subdir', ''))

            downloads.add((url, dest))

    config_references = [expand_path(config_ref) for config_ref in get_all_elems_from_json(config, 'config_path')]

    downloads |= {(url, dest) for config in config_references for url, dest in get_config_downloads(config)}

    return downloads


def get_configs_downloads(config: Optional[Union[str, Path, dict]] = None) -> Dict[str, Set[Path]]:
    all_downloads = defaultdict(set)
    if config:
        configs = [config]
    else:
        configs = list(Path(deeppavlov.__path__[0], 'configs').glob('**/*.json'))

    for config in configs:
        for url, dest in get_config_downloads(config):
            all_downloads[url].add(dest)

    return all_downloads


def check_md5(url: str, dest_paths: List[Path], headers: Optional[dict] = None) -> bool:
    url_md5 = path_set_md5(url)

    try:
        if url_md5.startswith('s3://'):
            import boto3

            s3 = boto3.resource('s3')
            bucket, key = url_md5[5:].split('/', maxsplit=1)
            obj = s3.Object(bucket, key)
            data = obj.get()['Body'].read().decode('utf8')
        else:
            r = requests.get(url_md5, headers=headers)
            if r.status_code != 200:
                return False
            data = r.text
    except Exception as e:
        log.debug(f'Could not download {url_md5} because of an exception {type(e)}: {e}')
        return False

    expected = {}
    for line in data.splitlines():
        _md5, fname = line.split(' ', maxsplit=1)
        if fname[0] != '*':
            if fname[0] == ' ':
                log.warning(f'Hash generated in text mode for {fname}, comparison could be incorrect')
            else:
                log.error(f'Unknown hash content format in {url + ".md5"}')
                return False
        expected[fname[1:]] = _md5

    done = None
    not_done = []
    for base_path in dest_paths:
        if all(file_md5(base_path / p) == _md5 for p, _md5 in expected.items()):
            done = base_path
        else:
            not_done.append(base_path)

    if done is None:
        return False

    for base_path in not_done:
        log.info(f'Copying data from {done} to {base_path}')
        for p in expected.keys():
            shutil.copy(done / p, base_path / p)
    return True


def download_resource(url: str, dest_paths: Iterable[Union[Path, str]], headers: Optional[dict] = None) -> None:
    dest_paths = [Path(dest) for dest in dest_paths]
    download_path = dest_paths[0].parent
    download_path.mkdir(parents=True, exist_ok=True)
    file_name = urlparse(url).path.split('/')[-1]
    lockfile = download_path / f'.{file_name}.lock'

    with FileLock(lockfile).acquire(poll_intervall=10):
        if check_md5(url, dest_paths, headers):
            log.info(f'Skipped {url} download because of matching hashes')
        elif any(ext in url for ext in ('.tar.gz', '.gz', '.zip')):
            download_decompress(url, download_path, dest_paths, headers=headers)
        else:
            dest_files = [dest_path / file_name for dest_path in dest_paths]
            download(dest_files, url, headers=headers)


def download_resources(args: Namespace) -> None:
    if not args.all and not args.config:
        log.error('You should provide either skill config path or -all flag')
        sys.exit(1)
    elif args.all:
        downloads = get_configs_downloads()
    else:
        config_path = Path(args.config).resolve()
        downloads = get_configs_downloads(config=config_path)

    for url, dest_paths in downloads.items():
        download_resource(url, dest_paths)


def deep_download(config: Union[str, Path, dict]) -> None:
    downloads = get_configs_downloads(config)
    last_id = len(downloads) - 1
    session_id = secrets.token_urlsafe(32)

    for file_id, (url, dest_paths) in enumerate(downloads.items()):
        headers = {
            'dp-token': get_download_token(),
            'dp-session': session_id,
            'dp-file-id': str(last_id - file_id),
            'dp-version': deeppavlov.__version__
        }
        if not url.startswith('s3://') and not isinstance(config, dict):
            url = set_query_parameter(url, 'config', Path(config).stem)
        download_resource(url, dest_paths, headers)


def main(args: Optional[List[str]] = None) -> None:
    args = parser.parse_args(args)
    log.info("Downloading...")
    download_resources(args)
    log.info("\nDownload successful!")


if __name__ == "__main__":
    main()
