import csv
from pathlib import Path

import sys

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import is_done, download, mark_done
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('typos_wikipedia_reader')
class TyposWikipedia(DatasetReader):
    def __init__(self):
        pass

    @staticmethod
    def build(data_path: str):
        data_path = Path(data_path) / 'typos_wiki'

        fname = data_path / 'misspelings.tsv'

        if not is_done(data_path):
            url = 'https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines'

            download(fname, url)

            with fname.open() as f:
                data = []
                for line in f:
                    if line.strip().endswith('<pre>'):
                        break
                for line in f:
                    if line.strip().startswith('</pre>'):
                        break
                    data.append(line.strip().split('-&gt;'))

            with fname.open('w', newline='') as tsvfile:
                writer = csv.writer(tsvfile, delimiter='\t')
                for line in data:
                    writer.writerow(line)

            mark_done(data_path)

            print('Built', file=sys.stderr)
        return fname

    @staticmethod
    def read(data_path: str, *args, **kwargs):
        fname = TyposWikipedia.build(data_path)
        with fname.open(newline='') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            next(reader)
            res = [(mistake, correct) for mistake, correct in reader]
        return {'train': res}
