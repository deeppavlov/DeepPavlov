import datetime
import logging
from pathlib import Path
from shutil import rmtree
from typing import List, Optional

import requests
import yaml
from dateutil.parser import parse as parsedate

from deeppavlov import build_model
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.core.data.utils import simple_download
from deeppavlov.models.entity_linking.download_parse_utils.entities_parse import EntitiesParser
from deeppavlov.models.entity_linking.download_parse_utils.wikidata_parse import WikidataParser

log = logging.getLogger('spam_application')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

DATA_PATH = Path('/data').resolve()

WIKIDATA_PATH = DATA_PATH / 'wikidata.json.bz2'
WIKIDATA_URL = 'https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2'

PARSED_WIKIDATA_PATH = DATA_PATH / 'parsed_wikidata'
PARSED_WIKIDATA_OLD_PATH = DATA_PATH / 'parsed_wikidata_old'
PARSED_WIKIDATA_NEW_PATH = DATA_PATH / 'parsed_wikidata_new'

ENTITIES_PATH = DATA_PATH / 'entities'
ENTITIES_OLD_PATH = DATA_PATH / 'entities_old'
ENTITIES_NEW_PATH = DATA_PATH / 'entities_new'

FAISS_PATH = DATA_PATH / 'faiss'
FAISS_OLD_PATH = DATA_PATH / 'faiss_old'
FAISS_NEW_PATH = DATA_PATH / 'faiss_new'

STATE_PATH = Path('/home/ignatov/asdf/down/state.yaml').resolve()

LABELS_PATH = Path('/home/ignatov/asdf/down/labels.yaml').resolve()


class State:
    def __init__(self,
                 state_path: Path,
                 wikidata_created: Optional[str] = None,
                 wikidata_parsed: Optional[str] = None,
                 entities_wikidata: Optional[str] = None,
                 labels_updated: Optional[str] = None,
                 entities_parsed: Optional[str] = None,
                 faiss_updated: Optional[str] = None) -> None:
        self._state_path = state_path
        self.wikidata_created = wikidata_created
        self.wikidata_parsed = wikidata_parsed
        self.entities_wikidata = entities_wikidata
        self.labels_updated = labels_updated
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

    def entities_is_fresh(self, labels_mtime: str) -> bool:
        if self.entities_wikidata is None or self.labels_updated is None:
            return False
        return parsedate(self.entities_wikidata) >= parsedate(self.wikidata_parsed) \
               and parsedate(labels_mtime) <= parsedate(self.labels_updated)

    def faiss_is_fresh(self):
        if self.faiss_updated is None:
            return False
        return parsedate(self.faiss_updated) >= parsedate(self.entities_parsed)


class Labels:
    def __init__(self,
                 labels_path: Path = LABELS_PATH) -> None:
        self.labels_path = labels_path
        if self.labels_path.exists():
            with open(labels_path) as fin:
                self.mtime = str(datetime.datetime.fromtimestamp(self.labels_path.stat().st_mtime))
                self.labels = yaml.safe_load(fin)
        else:
            self.labels = []
            self.save()
            self.mtime = str(datetime.datetime.fromtimestamp(self.labels_path.stat().st_mtime))

    def add_label(self, label: str, entity_ids: List[str]) -> None:
        self.labels.append([label, entity_ids])

    def add_labels(self, labels_list: List[List]) -> None:
        self.labels.extend(labels_list)

    def save(self) -> None:
        with open(self.labels_path, 'w') as fout:
            yaml.dump(self.labels, fout)


def download_wikidata(state: State) -> None:
    r = requests.head(WIKIDATA_URL)
    remote_wikidata_created = r.headers['Last-Modified']

    if WIKIDATA_PATH.exists() and state.wikidata_is_fresh(remote_wikidata_created):
        log.info('Current wikidata is the latest. Downloading of wikidata is skipped')
    else:
        WIKIDATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        simple_download(WIKIDATA_URL, WIKIDATA_PATH)
        state.wikidata_created = remote_wikidata_created
        state.save()
        log.info('Wikidata updated')


def safe_rmtree(dir_path: Path):
    if dir_path.exists():
        rmtree(dir_path)


def parse_wikidata(state: State) -> None:
    if state.parsed_wikidata_is_fresh():
        log.info('Current parsed wikidata is the latest. Skipping parsing of wikidata')
    else:
        safe_rmtree(PARSED_WIKIDATA_NEW_PATH)
        wikidata_parser = WikidataParser(WIKIDATA_PATH,
                                         save_path=PARSED_WIKIDATA_NEW_PATH)
        wikidata_parser.parse()
        safe_rmtree(PARSED_WIKIDATA_OLD_PATH)
        if PARSED_WIKIDATA_PATH.exists():
            PARSED_WIKIDATA_PATH.rename(PARSED_WIKIDATA_OLD_PATH)
        PARSED_WIKIDATA_NEW_PATH.rename(PARSED_WIKIDATA_PATH)
        state.wikidata_parsed = state.wikidata_created
        state.save()


def parse_entities(state: State) -> None:
    labels = Labels()
    if state.entities_is_fresh(labels.mtime):
        log.info('Current entities is the latest. Skipping entities parsing')
    else:
        safe_rmtree(ENTITIES_NEW_PATH)
        entities_parser = EntitiesParser(load_path=PARSED_WIKIDATA_PATH,
                                         save_path=ENTITIES_NEW_PATH,
                                         filter_tags=False)  # !!!!!!!!!!!!!!!!!!!! Не забудь удалить
        entities_parser.load()
        entities_parser.parse()
        labels = Labels()  # entities parsing is quiet long, labels could be updated in this time, so variable is
                           # re-initialized intentionally
        for label, entity_ids in labels.labels:
            entities_parser.add_label(label, entity_ids)
        entities_parser.save()
        safe_rmtree(ENTITIES_OLD_PATH)
        if ENTITIES_PATH.exists():
            ENTITIES_PATH.rename(ENTITIES_OLD_PATH)
        ENTITIES_NEW_PATH.rename(ENTITIES_PATH)

        state.entities_wikidata = state.wikidata_parsed
        state.labels_updated = labels.mtime
        state.entities_parsed = str(datetime.datetime.now())
        state.save()


def update_faiss(state: State):
    if state.faiss_is_fresh():
        log.info('skipping faiss update')
    else:
        config = parse_config('entity_linking_vx_sep')
        config['chainer']['pipe'][-1]['load_path'] = config['chainer']['pipe'][-1]['save_path'] = str(ENTITIES_PATH)
        config['chainer']['pipe'][-1]['fit_vectorizer'] = True
        config['chainer']['pipe'][-1]['vectorizer_filename'] = FAISS_NEW_PATH / Path(config['chainer']['pipe'][-1]['vectorizer_filename']).name
        config['chainer']['pipe'][-1]['faiss_index_filename'] = FAISS_NEW_PATH / Path(config['chainer']['pipe'][-1]['faiss_index_filename']).name
        build_model(config)
        safe_rmtree(FAISS_OLD_PATH)
        if FAISS_PATH.exists():
            FAISS_PATH.rename(FAISS_OLD_PATH)
        FAISS_NEW_PATH.rename(FAISS_PATH)

        state.faiss_updated = state.entities_parsed
        state.save()
