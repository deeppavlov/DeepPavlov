import pkgutil
import json

import deeppavlov
from deeppavlov.core.common.registry import _registry_path, _REGISTRY


if __name__ == '__main__':
    _REGISTRY.clear()
    for _, pkg_name, _ in pkgutil.walk_packages(deeppavlov.__path__, deeppavlov.__name__+'.'):
        __import__(pkg_name)

    with _registry_path.open('w', encoding='utf-8') as f:
        json.dump(dict(sorted(_REGISTRY.items())), f, indent=2)
