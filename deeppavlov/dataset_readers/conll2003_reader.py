from pathlib import Path

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress


@register('conll2003_reader')
class Conll2003DatasetReader(DatasetReader):
    """Class to read training datasets in CoNLL-2003 format"""

    def read(self,
             data_path: str,
             dataset_name: str = None,
             provide_pos: bool = False,
             provide_context: bool = False,
             context_size: int = 3):
        self.provide_pos = provide_pos
        self.provide_context = provide_context
        self.context_size = context_size
        data_path = Path(data_path)
        files = list(data_path.glob('*.txt'))
        if 'train.txt' not in {file_path.name for file_path in files}:
            if dataset_name == 'conll2003':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/conll2003_v2.tar.gz'
            elif dataset_name == 'collection_rus':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/collection5.tar.gz'
            elif dataset_name == 'ontonotes':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/ontonotes_ner.tar.gz'
            else:
                raise RuntimeError('train.txt not found in "{}"'.format(data_path))
            data_path.mkdir(exist_ok=True, parents=True)
            download_decompress(url, data_path)
            files = list(data_path.glob('*.txt'))
        dataset = {}

        for file_name in files:
            name = file_name.with_suffix('').name
            dataset[name] = self.parse_ner_file(file_name)
        return dataset

    def parse_ner_file(self, file_name: Path):

        def add_context(samples, left, right=None, max_size=self.context_size):
            def get_tokens(sample, with_pos=self.provide_pos):
                return sample[0][0] if with_pos else sample[0]

            for i in range(left, right or 0):
                x, y = samples[i]
                l = max(left, i - max_size)
                r = min(right or (i + max_size + 1), i + max_size + 1)
                l_context = [get_tokens(s) for s in samples[l:i]]
                r_context = [get_tokens(s) for s in samples[i+1:r]]
                new_x = x if self.provide_pos else (x,)
                new_x += (l_context, r_context,)
                samples[i] = (new_x, y)
                
        samples = []
        with file_name.open(encoding='utf8') as f:
            tokens = []
            pos_tags = []
            tags = []
            cur_doc_size = 0
            for line in f:
                # Check end of the document
                if 'DOCSTART' in line:
                    if len(tokens) > 1:
                        if self.provide_pos:
                            samples.append(((tokens, pos_tags), tags,))
                        else:
                            samples.append((tokens, tags,))
                        cur_doc_size += 1
                        if self.provide_context:
                            add_context(samples, -cur_doc_size)

                        tokens = []
                        pos_tags = []
                        tags = []
                        cur_doc_size = 0
                elif len(line) < 2:
                    if len(tokens) > 0:
                        if self.provide_pos:
                            samples.append(((tokens, pos_tags), tags,))
                        else:
                            samples.append((tokens, tags,))
                        tokens = []
                        pos_tags = []
                        tags = []
                        cur_doc_size += 1
                else:
                    if self.provide_pos:
                        token, pos, *_, tag = line.split()
                        pos_tags.append(pos)
                    else:
                        token, *_, tag = line.split()
                    tags.append(tag)
                    tokens.append(token)

            if self.provide_pos:
                samples.append(((tokens, pos_tags), tags,))
            else:
                samples.append((tokens, tags,))
            cur_doc_size += 1
            if self.provide_context:
                add_context(samples, -cur_doc_size)

        return samples
