import numpy as np
from pathlib import Path
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.inferable import Inferable


@register('dict_emb')
class DictEmbedder(Inferable):
    def __init__(self, ser_path, dim, *args, **kwargs):
        super().__init__(ser_path=ser_path)
        self.tok2emb = {}
        self.dim = dim

        self.load()

    def load(self):
        """
        Load dictionary of embeddings from file.
        """

        if not Path(self.ser_path).exists():
            raise FileNotFoundError(
                'There is no dictionary of embeddings <<{}>> file provided.'.format(
                    self.ser_path))
        else:
            print('Loading existing dictionary of embeddings from {}'.format(self.ser_path))

            with open(str(self.ser_path)) as fin:
                for line in fin:
                    values = line.rsplit(sep=' ', maxsplit=self.dim)
                    assert (len(values) == self.dim + 1)
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.tok2emb[word] = coefs

    @overrides
    def infer(self, sentence: str, *args, **kwargs) -> list:
        """
        Method returns embedded sentence
        Args:
            sentence: string (e.g. "I want some food")

        Returns:
            embedded sentence
        """
        return [self.tok2emb[t] for t in sentence.split()]
