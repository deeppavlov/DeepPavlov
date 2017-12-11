from pathlib import Path

from deeppavlov.core.common import paths

from deeppavlov.core.agent.agent import Agent
from deeppavlov.core.common.file import read_json

USR_DIR = 'USR_DIR'


def set_usr_dir(config_path: str, usr_dir_name):
    # make a serialization user dir
    root_ = Path(config_path).resolve().parent
    usr_dir_path = root_.joinpath(usr_dir_name)
    if not usr_dir_path.exists():
        usr_dir_path.mkdir()
    paths.USR_PATH = usr_dir_path
    return usr_dir_path


def set_vocab_path():
    return paths.USR_PATH.joinpath('vocab.txt')


def build_agent_from_config(config_path: str):
    config = read_json(config_path)
    skill_configs = config['skills']
    commutator_config = config['commutator']
    return Agent(skill_configs, commutator_config)
