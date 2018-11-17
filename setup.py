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

from setuptools import setup, find_packages
import os
import re

import deeppavlov

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


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
    return re.sub(r']\((?!https?://)', r'](https://github.com/deepmipt/DeepPavlov/blob/master/', text)


setup(
    name='deeppavlov',
    packages=find_packages(exclude=('tests', 'docs')),
    version=deeppavlov.__version__,
    description=deeppavlov.__description__,
    long_description=readme(),
    long_description_content_type='text/markdown',
    author=deeppavlov.__author__,
    author_email=deeppavlov.__email__,
    license=deeppavlov.__license__,
    url='https://github.com/deepmipt/DeepPavlov',
    download_url='https://github.com/deepmipt/DeepPavlov/archive/' + deeppavlov.__version__ + '.tar.gz',
    keywords=deeppavlov.__keywords__,
    include_package_data=True,
    extras_require={
            'tests': [
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
