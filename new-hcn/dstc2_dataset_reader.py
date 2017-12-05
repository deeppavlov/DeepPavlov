import logging
import itertools
from overrides import overrides
from pathlib import Path

from deeppavlov.common.registry import register_model
from deeppavlov.common import paths
from deeppavlov.data.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)


@register_model('dstc2')
class DSTC2DatasetReader(DatasetReader):

    @overrides
    def read(self, file_path):
        logger.info("Reading instances from lines in file at: {}".format(file_path))
        utterances, responses, dialog_indices =\
                self._read_turns(file_path, with_indices=True)

        responses_path = Path(paths.USR_PATH).joinpath('responses.txt')
        responses_path.write_text('\n'.join(responses))

        data = [ {'context': u, 'response': r}\
                 for u, r in zip(utterances, responses) ]

        return [ data[idx['start']:idx['end']] for idx in dialog_indices ]

    @staticmethod
    def _read_turns(file_path, with_indices=False):
        def _filter(turn):
            del turn['index']
            return turn

        utterances = []
        responses = []
        dialog_indices = []
        n = 0
        num_dial_utter, num_dial_respo = 0, 0
        for ln in open(file_path, 'rt'):
            if not ln.strip():
                assert len(num_dial_utter) == len(num_resp_utter),\
                        "Datafile in the wrong format."
                n += num_dial_utter
                dialog_indices.append({
                    'start': n - num_dial_utter,
                    'end': n,
                })
                num_dial_utter, num_dial_resp = 0, 0
            else:
                replica = _filter(json.loads(ln))
                if 'goals' in replica:
                    utterances.append(replica)
                    num_dial_utter += 1
                else:
                    responses.append(replica)
                    num_dial_resp += 1

        if with_indices:
            return utterances, responses, dialog_indices
        return utterances, responses

    @staticmethod
    @overrides
    def save_vocab(dialogs, fpath):
        with open(fpath, 'wt') as f:
            words = sorted(set(itertools.chain.from_iterable(
                turn['context'].lower().split()\
                for dialog in dialogs for turn in dialog
            )))
            f.write(' '.join(words))
