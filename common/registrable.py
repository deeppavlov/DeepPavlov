from collections import defaultdict
from typing import Dict, List, Type, TypeVar, Optional

from common.errors import ConfigError

T = TypeVar('T')


class Registrable:
    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)

    @classmethod
    def register(cls: Type[T], name: str):
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]):
            if name in registry:
                message = 'Cannot register `{0}` as {0}, name already in use for {1}' \
                    .format(name, cls.__name__, registry[name].__name__)
                raise ConfigError(message)
            registry[name] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) -> Optional[type]:
        if name not in Registrable._registry[cls]:
            raise ConfigError('`{}` is not a registered name for {}'.format(name, cls.__name__))
        return Registrable._registry[cls].get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        keys = list(Registrable._registry[cls].keys())
        return [k for k in keys if k]
