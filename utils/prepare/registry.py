import pkgutil
import json

import deeppavlov
from deeppavlov.core.common.registry import _registry_path as c_registry_path, _REGISTRY as C_REGISTRY
from deeppavlov.core.common.metrics_registry import _registry_path as m_registry_path, _REGISTRY as M_REGISTRY


if __name__ == '__main__':
    C_REGISTRY.clear()
    M_REGISTRY.clear()

    for _, pkg_name, _ in pkgutil.walk_packages(deeppavlov.__path__, deeppavlov.__name__+'.'):
        __import__(pkg_name)

    with c_registry_path.open('w', encoding='utf-8') as f:
        json.dump(dict(sorted(C_REGISTRY.items())), f, indent=2)

    with m_registry_path.open('w', encoding='utf-8') as f:
        json.dump(dict(sorted(M_REGISTRY.items())), f, indent=2)
