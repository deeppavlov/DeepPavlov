# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

from setuptools import setup, find_packages

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


__version__ = '0.9.1'
__author__ = 'Neural Networks and Deep Learning lab, MIPT'
__description__ = 'An open source library for building end-to-end dialog systems and training chatbots.'
__keywords__ = ['NLP', 'NER', 'SQUAD', 'Intents', 'Chatbot']
__license__ = 'Apache License, Version 2.0'
__email__ = 'info@deeppavlov.ai'


def create_meta():
    meta_path = os.path.join(__location__, 'deeppavlov', '_meta.py')
    with open(meta_path, 'w') as meta:
        meta.writelines([
            f'__version__ = {repr(__version__)}\n',
            f'__author__ = {repr(__author__)}\n',
            f'__description__ = {repr(__description__)}\n',
            f'__keywords__ = {repr(__keywords__)}\n',
            f'__license__ = {repr(__license__)}\n',
            f'__email__ = {repr(__email__)}\n'
        ])


def read_requirements():
    """parses requirements from requirements.txt"""
    reqs_path = os.path.join(__location__, 'requirements.txt')
    with open(reqs_path, encoding='utf8') as f:
        reqs = [line.strip() for line in f if not line.strip().startswith('#')]

    names = []
    links = []
    for req in reqs:
        if '://' in req:
            links.append(req)
        else:
            names.append(req)
    return {'install_requires': names, 'dependency_links': links}


def readme():
    with open(os.path.join(__location__, 'README.md'), encoding='utf8') as f:
        text = f.read()
    text = re.sub(r']\((?!https?://)', r'](https://github.com/deepmipt/DeepPavlov/blob/master/', text)
    text = re.sub(r'\ssrc="(?!https?://)', r' src="https://raw.githubusercontent.com/deepmipt/DeepPavlov/master/', text)
    return text


if __name__ == '__main__':
    create_meta()
    setup(
        name='deeppavlov',
        packages=find_packages(exclude=('tests', 'docs', 'utils')),
        version=__version__,
        description=__description__,
        long_description=readme(),
        long_description_content_type='text/markdown',
        author=__author__,
        author_email=__email__,
        license=__license__,
        url='https://github.com/deepmipt/DeepPavlov',
        download_url=f'https://github.com/deepmipt/DeepPavlov/archive/{__version__}.tar.gz',
        keywords=__keywords__,
        include_package_data=True,
        extras_require={
            'tests': [
                'flake8',
                'pytest',
                'pexpect'],
            'docs': [
                'sphinx>=1.7.9',
                'sphinx_rtd_theme>=0.4.0',
                'nbsphinx>=0.3.4',
                'ipykernel>=4.8.0'
            ]},
        **read_requirements()
    )
