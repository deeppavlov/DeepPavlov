from pathlib import Path

import numpy as np

from deeppavlov.core.common import paths
from deeppavlov.core.common.registry import register
from deeppavlov.models.trackers.hcn_et import EntityTracker

'''
    Action Templates

    1. 'any preference on a type of cuisine',
    2. 'api_call <party_size> <rest_type>',
    3. 'great let me do the reservation',
    4. 'hello what can i help you with today',
    5. 'here it is <info_address>',
    6. 'here it is <info_phone>',
    7. 'how many people would be in your party',
    8. "i'm on it",
    9. 'is there anything i can help you with',
    10. 'ok let me look into some options for you',
    11. 'sure is there anything else to update',
    12. 'sure let me find an other option for you',
    13. 'what do you think of this option: ',
    14. 'where should it be',
    15. 'which price range are looking for',
    16. "you're welcome",

    [1] : cuisine
    [2] : location
    [3] : party_size
    [4] : rest_type

'''


@register('hcn_at')
class ActionTracker:
    def __init__(self, ent_tracker=EntityTracker):
        # maintain an instance of EntityTracker
        self.et = ent_tracker
        # get a list of action templates
        self.action_templates = self.get_action_templates(Path(paths.deeppavlov_root) / 'responses.txt')
        self.action_size = len(self.action_templates)
        # action mask
        self.am = np.zeros([self.action_size], dtype=np.float32)
        # action mask lookup, built on intuition
        self.am_dict = {
            '0000': [4, 8, 1, 14, 7, 15],
            '0001': [4, 8, 1, 14, 7],
            '0010': [4, 8, 1, 14, 15],
            '0011': [4, 8, 1, 14],
            '0100': [4, 8, 1, 7, 15],
            '0101': [4, 8, 1, 7],
            '0110': [4, 8, 1, 15],
            '0111': [4, 8, 1],
            '1000': [4, 8, 14, 7, 15],
            '1001': [4, 8, 14, 7],
            '1010': [4, 8, 14, 15],
            '1011': [4, 8, 14],
            '1100': [4, 8, 7, 15],
            '1101': [4, 8, 7],
            '1110': [4, 8, 15],
            '1111': [2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 16]
        }

    def action_mask(self):
        # get context features as string of ints (0/1)
        ctxt_f = ''.join([str(flag) for flag in self.et.context_features().astype(np.int32)])

        def construct_mask(ctxt_f):
            indices = self.am_dict[ctxt_f]
            for index in indices:
                self.am[index - 1] = 1.
            return self.am

        return construct_mask(ctxt_f)

    def get_action_templates(self, responses_path: Path):
        responses = responses_path.read_text().split('\n')
        responses = list(set([self.et._extract_entities(response, update=False)
                              for response in responses]))

        # extract restaurant entities
        return sorted(set([self._extract(response) for response in responses]))

    def get_template_id(self, response):

        return self.action_templates.index(
            self._extract(self.et._extract_entities(response, update=False))
        )

    @staticmethod
    def _extract(response):
        template = []
        for word in response.split(' '):
            if 'resto_' in word:
                if 'phone' in word:
                    template.append('<info_phone>')
                elif 'address' in word:
                    template.append('<info_address>')
                else:
                    template.append('<restaurant>')
            else:
                template.append(word)
        return ' '.join(template)

    def reset(self, ent_tracker):
        self.et = ent_tracker
        self.am = np.zeros([self.action_size], dtype=np.float32)



