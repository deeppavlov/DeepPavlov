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

from setuptools import setup, find_packages
import os
try:  # for pip>=10.0.0
    from pip._internal.req import parse_requirements
    from pip._internal.download import PipSession
    from pip._internal import main as pip_main
except ImportError:  # for pip<=9.0.3
    from pip.req import parse_requirements
    from pip.download import PipSession
    from pip import main as pip_main

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def read_requirements():
    # # parses requirements from requirements.txt
    reqs_path = os.path.join(__location__, 'requirements.txt')
    install_reqs = parse_requirements(reqs_path, session=PipSession())
    reqs = []
    for ir in install_reqs:
        pip_main(['install', str(ir.req or ir.link)])
        if ir.req:
            reqs.append(str(ir.req))
    return reqs


def readme():
    with open(os.path.join(__location__, 'README.md')) as f:
        return f.read()


setup(
    name='deeppavlov',
    packages=find_packages(exclude=('tests',)),
    version='0.0.4',
    description='An open source library for building end-to-end dialog systems and training chatbots.',
    long_description=readme(),
    long_description_content_type="text/markdown",
    author='Neural Networks and Deep Learning lab, MIPT',
    author_email='info@ipavlov.ai',
    license='Apache License, Version 2.0',
    url='https://github.com/deepmipt/DeepPavlov',
    download_url='https://github.com/deepmipt/DeepPavlov/archive/0.0.4.tar.gz',
    keywords=['NLP', 'NER', 'SQUAD', 'Intents', 'Chatbot'],
    include_package_data=True,
    install_requires=read_requirements()
)
