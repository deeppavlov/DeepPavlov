import datetime
import logging
from pathlib import Path
from shutil import rmtree, copytree
from typing import Optional

import yaml
from dateutil.parser import parse as parsedate

from aliases import Aliases
from constants import WIKIDATA_PATH, WIKIDATA_URL, PARSED_WIKIDATA_PATH, PARSED_WIKIDATA_OLD_PATH, \
    PARSED_WIKIDATA_NEW_PATH, ENTITIES_PATH, ENTITIES_OLD_PATH, ENTITIES_NEW_PATH, FAISS_PATH, FAISS_OLD_PATH, \
    FAISS_NEW_PATH, STATE_PATH, DATA_PATH, LOGS_PATH
from deeppavlov import build_model
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.core.data.utils import simple_download
from deeppavlov.models.entity_linking.download_parse_utils.entities_parse import EntitiesParser
from deeppavlov.models.entity_linking.download_parse_utils.wikidata_parse import WikidataParser

log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


class State:
    def __init__(self,
                 state_path: Path,
                 wikidata_created: Optional[str] = None,
                 wikidata_parsed: Optional[str] = None,
                 entities_wikidata: Optional[str] = None,
                 aliases_updated: Optional[datetime.datetime] = None,
                 entities_parsed: Optional[str] = None,
                 faiss_updated: Optional[str] = None) -> None:
        self._state_path = state_path
        self.wikidata_created = wikidata_created
        self.wikidata_parsed = wikidata_parsed
        self.entities_wikidata = entities_wikidata
        self.aliases_updated = aliases_updated
        self.entities_parsed = entities_parsed
        self.faiss_updated = faiss_updated

    def save(self) -> None:
        with open(self._state_path, 'w') as fout:
            yaml.dump({k: v for k, v in self.__dict__.items() if not k.startswith('_')}, fout)

    @classmethod
    def from_yaml(cls, state_path: Path = STATE_PATH):
        if state_path.exists():
            with open(state_path) as fin:
                params = yaml.safe_load(fin)
        else:
            params = {}
        return cls(state_path, **params)

    def wikidata_is_fresh(self, remote_wikidata_created: str) -> bool:
        if self.wikidata_created is None:
            return False
        return parsedate(remote_wikidata_created) == parsedate(self.wikidata_created)

    def parsed_wikidata_is_fresh(self) -> bool:
        if self.wikidata_parsed is None:
            return False
        return parsedate(self.wikidata_created) <= parsedate(self.wikidata_parsed)

    def entities_is_fresh(self, aliases_mtime: datetime.datetime) -> bool:
        if self.entities_wikidata is None or self.aliases_updated is None:
            return False
        return parsedate(self.entities_wikidata) >= parsedate(self.wikidata_parsed) \
               and aliases_mtime <= self.aliases_updated

    def faiss_is_fresh(self):
        if self.faiss_updated is None:
            return False
        return parsedate(self.faiss_updated) >= parsedate(self.entities_parsed)


def download_wikidata() -> None:
    print('Wikidate update started', flush=True)
    simple_download(WIKIDATA_URL, WIKIDATA_PATH)
    print('Wikidata updated', flush=True)


def safe_rmtree(dir_path: Path):
    if dir_path.exists():
        rmtree(dir_path)


def parse_wikidata() -> None:
    print('Wikidata parsing started', flush=True)
    safe_rmtree(PARSED_WIKIDATA_NEW_PATH)
    wikidata_parser = WikidataParser(WIKIDATA_PATH,
                                     save_path=PARSED_WIKIDATA_NEW_PATH)
    wikidata_parser.parse()
    safe_rmtree(PARSED_WIKIDATA_OLD_PATH)
    if PARSED_WIKIDATA_PATH.exists():
        PARSED_WIKIDATA_PATH.rename(PARSED_WIKIDATA_OLD_PATH)
    PARSED_WIKIDATA_NEW_PATH.rename(PARSED_WIKIDATA_PATH)
    print('Wikidata parsed', flush=True)


def parse_entities() -> None:
    print('Entities parsing started', flush=True)
    safe_rmtree(ENTITIES_NEW_PATH)
    ENTITIES_NEW_PATH.mkdir(parents=True, exist_ok=True)
    entities_parser = EntitiesParser(load_path=PARSED_WIKIDATA_PATH,
                                     save_path=ENTITIES_NEW_PATH)
    entities_parser.load()
    log.info("----- loaded parser")
    entities_parser.parse()
    log.info("----- parsed")
    aliases = Aliases()
    for label, entity_ids in aliases.aliases.items():
        entities_parser.add_label(label, entity_ids)
    entities_parser.save()
    safe_rmtree(ENTITIES_OLD_PATH)
    if ENTITIES_PATH.exists():
        ENTITIES_PATH.rename(ENTITIES_OLD_PATH)
    ENTITIES_NEW_PATH.rename(ENTITIES_PATH)
    print('Entities parsing finished', flush=True)


def update_faiss():
    print('Faiss update started', flush=True)
    safe_rmtree(FAISS_NEW_PATH)
    FAISS_NEW_PATH.mkdir(parents=True, exist_ok=True)
    config = parse_config('entity_linking_vx_siam_distil.json')
    config['chainer']['pipe'][-1]['load_path'] = config['chainer']['pipe'][-1]['save_path'] = str(ENTITIES_PATH)
    config['chainer']['pipe'][-1]['fit_tfidf_vectorizer'] = True
    config['chainer']['pipe'][-1]['fit_fasttext_vectorizer'] = True
    config['chainer']['pipe'][-1]['fasttext_vectorizer_filename'] = \
        FAISS_NEW_PATH / Path(config['chainer']['pipe'][-1]['fasttext_vectorizer_filename']).name
    config['chainer']['pipe'][-1]['tfidf_vectorizer_filename'] = \
        FAISS_NEW_PATH / Path(config['chainer']['pipe'][-1]['tfidf_vectorizer_filename']).name
    config['chainer']['pipe'][-1]['fasttext_faiss_index_filename'] = \
        FAISS_NEW_PATH / Path(config['chainer']['pipe'][-1]['fasttext_faiss_index_filename']).name
    config['chainer']['pipe'][-1]['tfidf_faiss_index_filename'] = \
        FAISS_NEW_PATH / Path(config['chainer']['pipe'][-1]['tfidf_faiss_index_filename']).name
    build_model(config)
    log.info('faiss cpu updated')
    safe_rmtree(FAISS_OLD_PATH)
    if FAISS_PATH.exists():
        FAISS_PATH.rename(FAISS_OLD_PATH)
    FAISS_NEW_PATH.rename(FAISS_PATH)
    print('Faiss update finished')


def initial_setup():
    if not ENTITIES_PATH.exists():
        copytree(f'{DATA_PATH}/downloads/entities', ENTITIES_PATH)
    if not FAISS_PATH.exists():
        copytree(f'{DATA_PATH}/downloads/faiss', FAISS_PATH)
    if not PARSED_WIKIDATA_PATH.exists():
        copytree(f'{DATA_PATH}/downloads/parsed_wikidata', PARSED_WIKIDATA_PATH)
    if not LOGS_PATH.exists():
        LOGS_PATH.mkdir(parents=True)
