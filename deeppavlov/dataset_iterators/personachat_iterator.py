from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_iterator import BasicDatasetIterator


@register('personachat_iterator')
class PersonaChatIterator(BasicDatasetIterator):
    def split(self, *args, **kwargs):
        for dt in ['train', 'valid', 'test']:
            setattr(self, dt, PersonaChatIterator._to_tuple(getattr(self, dt)))

    @staticmethod
    def _to_tuple(data):
        """
        Returns:
            list of (persona, x, candidates), y
        """
        return list(map(lambda x: ((x['persona'], x['x'], x['candidates']), x['y']), data))
