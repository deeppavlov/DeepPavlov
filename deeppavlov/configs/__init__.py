from pathlib import Path
from typing import Iterator, Dict, Union, Iterable


class Struct:
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

        self.keys = lambda: self._keys

    def _asdict(self, *, to_string: bool=False) -> dict:
        res = []
        for key in self._keys:
            value = getattr(self, key)
            if isinstance(value, Struct):
                value = value._asdict(to_string=to_string)
            elif to_string:
                value = str(value)
            res.append((key, value))

        return dict(res)

    def __getitem__(self, key: str) -> Union[dict, Path]:
        if key not in self._keys:
            raise KeyError(key)

        item = getattr(self, key)
        if isinstance(item, Struct):
            item = item._asdict()
        return item

    def __dir__(self) -> Iterable:
        return self._keys

    def _ipython_key_completions_(self) -> Iterable:
        return self._keys

    def __str__(self) -> str:
        return str(self._asdict(to_string=True))

    def __repr__(self) -> str:
        return f'Struct({repr(self._asdict())})'

    def _repr_pretty_(self, p, cycle):
        """method that defines ``Struct``'s pretty printing rules for iPython

        Args:
            p (IPython.lib.pretty.RepresentationPrinter): pretty printer object
            cycle (bool): is ``True`` if pretty detected a cycle
        """
        if cycle:
            p.text('Struct(...)')
        else:
            with p.group(7, 'Struct(', ')'):
                p.pretty(self._asdict())


def _build_configs_tree() -> Struct:
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
