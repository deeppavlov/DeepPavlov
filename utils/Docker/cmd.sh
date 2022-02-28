#!/bin/bash

pip install .[tests,docs]

if [ $(python -c 'import sys; print(sys.version_info[1])') -le 7 ]
then
  pip install -r deeppavlov/requirements/tf-gpu.txt

rm -rf `find . -mindepth 1 -maxdepth 1 ! -name tests ! -name Jenkinsfile ! -name docs

cd docs
make clean
make html
cd ..

flake8 `python -c 'import deeppavlov; print(deeppavlov.__path__[0])'` --count --select=E9,F63,F7,F82 --show-source --statistics

pytest -v --disable-warnings
