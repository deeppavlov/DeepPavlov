from pathlib import Path

from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.common.registry import register


@register('personachat_dataset_reader')
class PersonaChatDatasetReader(DatasetReader):
    """
    PersonaChat dataset from
    Zhang S. et al. Personalizing Dialogue Agents: I have a dog, do you have pets too?
    https://arxiv.org/abs/1801.07243
    Also, this dataset is used by ConvAI2.

    reads dataset to the following format:
    [{
        'persona': [list of persona sentences],
        'x': input utterance,
        'y': output utterance,
        'candidates': [ list of candidate utterances]
      },
       ...
    ]

    """

    url = 'http://lnsigo.mipt.ru/export/datasets/personachat.tar.gz'

    def read(self, dir_path: str, mode='self_original'):
        assert mode in ['self_original', 'self_revised'], print('mode: {} is unsupported'.format(mode))
        dir_path = Path(dir_path)
        required_files = ['{}_{}.txt'.format(dt, mode) for dt in ['train', 'valid', 'test']]
        if not dir_path.exists():
            dir_path.mkdir()

        if not all((dir_path / f).exists() for f in required_files):
            download_decompress(self.url, dir_path)

        dataset = {}
        for dt in ['train', 'valid', 'test']:
            dataset[dt] = self._parse_data(dir_path / '{}_{}.txt'.format(dt, mode))

        return dataset

    @staticmethod
    def _parse_data(filename):
        examples = []
        print(filename)
        curr_persona = []
        persona_done = False
        with filename.open('r') as fin:
            for line in fin:
                line = ' '.join(line.strip().split(' ')[1:])
                your_persona_pref = 'your persona: '
                if line[:len(your_persona_pref)] == your_persona_pref and persona_done:
                    curr_persona = [line[len(your_persona_pref):]]
                    persona_done = False
                elif line[:len(your_persona_pref)] == your_persona_pref:
                    curr_persona.append(line[len(your_persona_pref):])
                else:
                    persona_done = True
                    x, y, _, candidates = line.split('\t')
                    example = {
                        'persona': curr_persona,
                        'x': x,
                        'y': y,
                        'candidates': candidates.split('|'),
                    }
                    examples.append(example)

        return examples
