import datetime
import logging
from pathlib import Path
from shutil import copy, rmtree
from typing import Optional

import requests
import yaml
from dateutil.parser import parse as parsedate

from constants import WIKIDATA_PATH, WIKIDATA_URL, PARSED_WIKIDATA_PATH, PARSED_WIKIDATA_OLD_PATH, \
    PARSED_WIKIDATA_NEW_PATH, ENTITIES_PATH, ENTITIES_OLD_PATH, ENTITIES_NEW_PATH, FAISS_PATH, FAISS_OLD_PATH, \
    FAISS_NEW_PATH, STATE_PATH
from aliases import Aliases
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


def download_wikidata(state: State) -> None:
    r = requests.head(WIKIDATA_URL)
    remote_wikidata_created = r.headers['Last-Modified']

    if WIKIDATA_PATH.exists() and state.wikidata_is_fresh(remote_wikidata_created):
        log.info('Current wikidata is the latest. Downloading of wikidata is skipped')
    else:
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
    aliases = Aliases()
    if state.entities_is_fresh(aliases.mtime):
        log.info('Current entities is the latest. Skipping entities parsing')
    else:
        safe_rmtree(ENTITIES_NEW_PATH)
        ENTITIES_NEW_PATH.mkdir(parents=True, exist_ok=True)
        entities_parser = EntitiesParser(load_path=PARSED_WIKIDATA_PATH,
                                         save_path=ENTITIES_NEW_PATH)
        entities_parser.load()
        entities_parser.parse()
        aliases = Aliases()  # entities parsing is quiet long, at this time labels could be updated,
        # so variable is re-initialized intentionally
        for label, entity_ids in aliases.aliases.items():
            entities_parser.add_label(label, entity_ids)
        entities_parser.save()
        safe_rmtree(ENTITIES_OLD_PATH)
        if ENTITIES_PATH.exists():
            ENTITIES_PATH.rename(ENTITIES_OLD_PATH)
        ENTITIES_NEW_PATH.rename(ENTITIES_PATH)

        state.entities_wikidata = state.wikidata_parsed
        state.aliases_updated = aliases.mtime
        state.entities_parsed = str(datetime.datetime.now())
        state.save()


def update_faiss(state: State):
    if state.faiss_is_fresh():
        log.info('skipping faiss update')
    else:
        safe_rmtree(FAISS_NEW_PATH)
        FAISS_NEW_PATH.mkdir(parents=True, exist_ok=True)
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


def initial_setup():
    if not ENTITIES_PATH.exists():
        ENTITIES_PATH.mkdir(parents=True)
        copy('/root/.deeppavlov/downloads/wikidata_rus/entities_ranking_dict_vx.pickle', ENTITIES_PATH / 'entities_ranking_dict_vx.pickle')
        copy('/root/.deeppavlov/downloads/wikidata_rus/entities_types_sets.pickle', ENTITIES_PATH / 'entities_types_sets.pickle')
        copy('/root/.deeppavlov/downloads/wikidata_rus/q_to_descr_vx.pickle', ENTITIES_PATH / 'q_to_descr_vx.pickle')
        copy('/root/.deeppavlov/downloads/wikidata_rus/word_to_idlist_vx.pickle', ENTITIES_PATH / 'word_to_idlist_vx.pickle')
    if not FAISS_PATH.exists():
        FAISS_PATH.mkdir(parents=True)
        copy('/root/.deeppavlov/downloads/wikidata_rus/vectorizer_vx.pk', FAISS_PATH / 'vectorizer_vx.pk')
        copy('/root/.deeppavlov/downloads/wikidata_rus/faiss_vectors_gpu.index', FAISS_PATH / 'faiss_vectors_gpu.index')
