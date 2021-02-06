from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Dict, List
from constants import ALIASES_PATH
import yaml

log = getLogger(__file__)


class Aliases:
    def __init__(self, aliases_path: Path = ALIASES_PATH) -> None:
        self.aliases_path = aliases_path
        if self.aliases_path.exists():
            with open(self.aliases_path) as fin:
                self.mtime = datetime.fromtimestamp(self.aliases_path.stat().st_mtime)
                self.aliases = yaml.safe_load(fin)
                assert isinstance(self.aliases, dict), f'file {self.aliases_path} contains {type(self.aliases)} ' \
                                                       f'instead of dict'
        else:
            self.aliases = {}
            self.save()

    def add_alias(self, label: str, entity_ids: List[str]) -> None:
        self.aliases[label] = entity_ids
        self.save()

    def add_aliases(self, aliases: Dict[str, List[str]]) -> None:
        self.aliases.update(aliases)
        self.save()

    def delete_alias(self, label: str):
        try:
            del self.aliases[label]
        except KeyError:
            pass
        self.save()

    def save(self) -> None:
        with open(self.aliases_path, 'w') as fout:
            yaml.dump(self.aliases, fout)
        self.mtime = datetime.fromtimestamp(self.aliases_path.stat().st_mtime)
