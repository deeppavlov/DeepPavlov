import sys
from pathlib import Path

import yaml
from mongoengine import connect

DB_NAME = 'test'
HOST = 'localhost'
PORT = 27017

_run_config_path: Path = Path(__file__).resolve().parent / 'config.yaml'
_module = sys.modules[__name__]

if _run_config_path.is_file():
    with _run_config_path.open('r', encoding='utf-8') as f:
        config: dict = yaml.safe_load(f)

    if config.get('use_config', False) is True:
        config = config.get('db_config', {})

        DB_NAME = config.get('DB_NAME', DB_NAME)
        HOST = config.get('HOST', HOST)
        PORT = config.get('PORT', PORT)

state_storage = connect(host=HOST, port=PORT, db=DB_NAME)
