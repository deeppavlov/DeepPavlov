#!/bin/bash

set -e

python -m deeppavlov install "${CONFIG}"

exec "$@"
