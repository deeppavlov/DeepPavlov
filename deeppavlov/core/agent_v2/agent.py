from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from typing import Collection, Hashable, Dict, List, Any, Optional, Union, Callable


class Agent:
    def __init__(self, states_manager, preprocessors: Dict[Callable, Union[str, Collection[Optional[str]]]], *,
                 max_workers: Optional[int] = None) -> None:
        self.states_manager = states_manager
        self.preprocessors = []
        self.keys = []
        for preprocessor, keys in preprocessors.items():
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
            self.preprocessors.append(preprocessor)

        self.executor = ThreadPoolExecutor(max_workers)

    def _predict_annotations(self, utterances: Collection[str]) -> List[Dict[str, Any]]:
        annotations = []
        for preprocessed in zip(*chain(*self.executor.map(lambda f: f(utterances), self.preprocessors))):
            dialog_annotations = {}
            for k, data in zip(self.keys, preprocessed):
                target = dialog_annotations
                for prefix in k[:-1]:
                    target = target.setdefault(prefix, {})
                target[k[-1]] = data

            annotations.append(dialog_annotations)

        return annotations

    def __call__(self, utterances: Collection[str], user_ids: Collection[Hashable]):
        should_reset = [utterance == '\\start' for utterance in utterances]
        dialog_states = self.states_manager.get(user_ids, should_reset)
        annotations = self._predict_annotations(utterances)
        ...
