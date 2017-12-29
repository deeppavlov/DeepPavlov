from pathlib import Path, PurePath

from deeppavlov.core.common import paths
from deeppavlov.core.common.file import read_json


def set_usr_dir(config_path: str, usr_dir_name='USR_DIR') -> PurePath:
    """
    Make a serialization user dir.
    """
    config = read_json(config_path)
    if 'usr_dir' in config:
        usr_dir = Path(config['usr_dir'])
    else:
        usr_dir = Path(config_path).expanduser().absolute().parent / usr_dir_name

    usr_dir.mkdir(mode=0o755, exist_ok=True)

    paths.USR_PATH = usr_dir
    return usr_dir


def set_vocab_path() -> PurePath:
    return paths.USR_PATH / 'vocab.txt'

