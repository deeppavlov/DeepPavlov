from pathlib import Path, PosixPath

from deeppavlov.core.common import paths
from deeppavlov.core.common.file import read_json


def set_usr_dir(config_path: str, usr_dir_name='USR_DIR') -> PosixPath:
    """
    Make a serialization user dir.
    """
    config = read_json(config_path)
    try:
        usr_dir = Path(config['usr_dir'])
    except KeyError:
        usr_dir = Path(config_path).expanduser().absolute().parent / usr_dir_name

    usr_dir.mkdir(exist_ok=True)

    paths.USR_PATH = usr_dir
    return usr_dir


