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

meta_path = os.path.join(__location__, 'deeppavlov', '_meta.py')
with open(meta_path) as meta:
    exec(meta.read())


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
    text = re.sub(r']\((?!https?://)', r'](https://github.com/deeppavlov/DeepPavlov/blob/master/', text)
    text = re.sub(r'\ssrc="(?!https?://)', r' src="https://raw.githubusercontent.com/deeppavlov/DeepPavlov/master/', text)
    return text


if __name__ == '__main__':
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
        url='https://github.com/deeppavlov/DeepPavlov',
        download_url=f'https://github.com/deeppavlov/DeepPavlov/archive/{__version__}.tar.gz',
        keywords=__keywords__,
        include_package_data=True,
        extras_require={
            'tests': [
                'flake8',
                'pytest',
                'pytest-instafail',
                'pexpect'
            ],
            'docs': [
                'sphinx==3.5.4',
                'sphinx_rtd_theme==0.5.2',
                'nbsphinx==0.8.4',
                'ipykernel==5.5.4',
                'jinja2<=3.0.3',
                'sphinx-copybutton==0.5.0',
                'pandoc==2.2',
                'ipython_genutils==0.2.0'
            ],
            's3': [
                'boto3'
            ]
        },
        **read_requirements()
    )
