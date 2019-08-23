from pathlib import Path

import deeppavlov

from deeppavlov import build_model
from deeppavlov.core.common.file import read_json
from deeppavlov.utils.pip_wrapper import install_from_config


vectorizer = None


def get_vectorizer():
    global vectorizer
    if vectorizer:
        return vectorizer
    else:
        model_config = read_json(Path(deeppavlov.__path__[0]) / "configs/vectorizer/fasttext_vectorizer.json")
        install_from_config(model_config)
        vectorizer = build_model(model_config)
        return vectorizer
