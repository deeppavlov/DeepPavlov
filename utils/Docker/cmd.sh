#!/bin/bash

set -e

pip install .[tests,docs]

rm -rf `find . -mindepth 1 -maxdepth 1 ! -name tests ! -name Jenkinsfile ! -name docs`

cd docs
make clean
make html
cd ..

flake8 `python -c 'import deeppavlov; print(deeppavlov.__path__[0])'` --count --select=E9,F63,F7,F82 --show-source --statistics

pytest -v --disable-warnings --instafail $PYTEST_ARGS
