from collections.abc import Mapping
from pathlib import Path
from typing import Iterator, Dict, Union


class Struct(Mapping):
    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __init__(self, tree: Dict[str, Union[dict, Path]]) -> None:
        self._keys = set()
        for key, value in tree.items():
            key = key.replace('.', '_')
            self._keys.add(key)
            setattr(self, key,
                    Struct(value) if isinstance(value, dict) else value)
        self._keys = frozenset(self._keys)

    def _asdict(self) -> dict:
        return {key: value._asdict() if isinstance(value, Struct) else value
                for key, value in self.__dict__.items() if key in self._keys}

    def __getitem__(self, key: str):
        if key not in self._keys:
            raise KeyError(key)
        return self._asdict()[key]

    def __str__(self):
        return str(self._asdict())

    def __repr__(self):
        return f'Struct({repr(self._asdict())})'


def _build_configs_tree():
    root = Path(__file__).resolve().parent

    tree = {}

    for config in root.glob('**/*.json'):
        leaf = tree
        for part in config.relative_to(root).parent.parts:
            if part not in leaf:
                leaf[part] = {}
            leaf = leaf[part]
        leaf[config.stem] = config

    return Struct(tree)


configs = _build_configs_tree()
