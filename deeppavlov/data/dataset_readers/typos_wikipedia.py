import csv
import os

from deeppavlov.common.registry import register_model
from deeppavlov.data.utils import is_done, download, mark_done
from .dataset_reader import DatasetReader


@register_model('typos_wikipedia_reader')
class TyposWikipedia(DatasetReader):
    @staticmethod
    def build(data_path: str):
        data_path = os.path.join(data_path, 'typos_wiki')

        fname = 'misspelings.tsv'
        fname = os.path.join(data_path, fname)

        if not is_done(data_path):
            url = 'https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines'

            download(fname, url)

            with open(fname) as f:
                data = []
                for line in f:
                    if line.strip().endswith('<pre>'):
                        break
                for line in f:
                    if line.strip().startswith('</pre>'):
                        break
                    data.append(line.strip().split('-&gt;'))

            with open(fname, 'w', newline='') as tsvfile:
                writer = csv.writer(tsvfile, delimiter='\t')
                for line in data:
                    writer.writerow(line)

            mark_done(data_path)

            print('Built')
        return fname

    @staticmethod
    def read(data_path: str, *args, **kwargs):
        fname = TyposWikipedia.build(data_path)
        with open(fname, newline='') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            next(reader)
            res = [(mistake, correct) for mistake, correct in reader]
        return {'train': res}
