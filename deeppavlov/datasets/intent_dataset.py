from sklearn.model_selection import train_test_split

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset import Dataset


@register('intent_dataset')
class IntentDataset(Dataset):
    """
    Class gets data dictionary from dataset_reader, merge fields if necessary,
    split a field if necessary,
    """
    def __init__(self, data,
                 seed=None,
                 fields_to_merge=None, merged_field=None,
                 field_to_split=None, split_fields=None, split_proportions=None,
                 *args, **kwargs):

        super().__init__(data, seed)
        self.classes = None

        new_data = dict()
        new_data['train'] = []
        new_data['valid'] = []
        new_data['test'] = []

        for field in ['train', 'valid', 'test']:
            for turn in self.data[field]:
                reply = turn[0]
                curr_intents = []
                if reply['intents']:
                    for intent in reply['intents']:
                        for slot in intent['slots']:
                            if slot[0] == 'slot':
                                curr_intents.append(intent['act'] + '_' + slot[1])
                            else:
                                curr_intents.append(intent['act'] + '_' + slot[0])
                        if len(intent['slots']) == 0:
                            curr_intents.append(intent['act'])
                else:
                    if reply['text']:
                        curr_intents.append('unknown')
                    else:
                        continue
                new_data[field].append((reply['text'], curr_intents))

        self.data = new_data

        if fields_to_merge is not None:
            if merged_field is not None:
                print("Merging fields <<{}>> to new field <<{}>>".format(fields_to_merge,
                                                                         merged_field))
                self._merge_data(fields_to_merge=fields_to_merge,
                                 merged_field=merged_field)
            else:
                raise IOError("Given fields to merge BUT not given name of merged field")

        if field_to_split is not None:
            if split_fields is not None:
                print("Splitting field <<{}>> to new fields <<{}>>".format(field_to_split,
                                                                           split_fields))
                self._split_data(field_to_split=field_to_split,
                                 split_fields=split_fields,
                                 split_proportions=[float(s) for s in
                                                    split_proportions])
            else:
                raise IOError("Given field to split BUT not given names of split fields")

    def _split_data(self, field_to_split, split_fields, split_proportions):
        data_to_div = self.data[field_to_split].copy()
        data_size = len(self.data[field_to_split])
        for i in range(len(split_fields) - 1):
            self.data[split_fields[i]], \
            data_to_div = train_test_split(data_to_div,
                                           test_size=
                                           len(data_to_div) - int(
                                               data_size * split_proportions[i]))
        self.data[split_fields[-1]] = data_to_div
        return True

    def _merge_data(self, fields_to_merge, merged_field):
        data = self.data.copy()
        data[merged_field] = []
        for name in fields_to_merge:
            data[merged_field] += self.data[name]
        self.data = data
        return True
