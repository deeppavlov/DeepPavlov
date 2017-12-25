from pathlib import Path, PurePath

from deeppavlov.core.common import paths
from deeppavlov.core.common.file import read_json


def set_usr_dir(config_path: str, usr_dir_name='USR_DIR') -> PurePath:
    """
    Make a serialization user dir.
    """
    config = read_json(config_path)
    try:
        usr_dir = Path(config['usr_dir'])
    except KeyError:
        parent = Path(config_path).resolve().parent
        usr_dir = parent.joinpath(usr_dir_name)

    if not usr_dir.exists():
        usr_dir.mkdir()

    paths.USR_PATH = usr_dir
    return usr_dir


def set_vocab_path() -> PurePath:
    return paths.USR_PATH.joinpath('vocab.txt')

