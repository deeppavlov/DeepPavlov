from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from typing import Collection, Dict, List, Any, Optional, Union, Callable


class Preprocessor:

    def __init__(self, annotators: Dict[Callable, Union[str, List[Optional[str]]]], *,
                 max_workers: Optional[int] = None) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.annotators = []
        self.keys = []
        for preprocessor, keys in annotators.items():
            if isinstance(keys, str):
                keys = [keys]

            if len(keys) == 1:
                old_preprocessor = preprocessor

                def preprocessor(x):
                    return [old_preprocessor(x)]
            elif None in keys:
                indexes, keys = zip(*[(i, k) for i, k in enumerate(keys) if k is not None])
                old_preprocessor = preprocessor

                def preprocessor(x):
                    res = old_preprocessor(x)
                    return [res[i] for i in indexes]

            self.keys.extend([k.split('.') for k in keys])
            self.annotators.append(preprocessor)

    def __call__(self, utterances: Collection[str]) -> List[Dict[str, Any]]:
        annotations = []
        for preprocessed in zip(*chain(*self.executor.map(lambda f: f(utterances), self.annotators))):
            dialog_annotations = {}
            for k, data in zip(self.keys, preprocessed):
                target = dialog_annotations
                for prefix in k[:-1]:
                    target = target.setdefault(prefix, {})
                target[k[-1]] = data

            annotations.append(dialog_annotations)

        return annotations
