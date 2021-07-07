import numpy as np
import pickle
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.utils import expand_path


@register('re_postprocessor')
class RePostprocessor:

    def __init__(self, rel2id_path: str, rel2label_path: str, return_rel_labels: bool = False, **kwargs):
        self.rel2id_path = rel2id_path
        self.rel2label_path = rel2label_path
        self.return_rel_labels = return_rel_labels
        self.rel2id = read_json(str(expand_path(self.rel2id_path)))
        self.id2rel = {rel_id: rel for rel, rel_id in self.rel2id.items()}
        self.rel2label = read_json(str(expand_path(self.rel2label_path)))
        
    def __call__(self, model_output):
        if self.return_rel_labels:
            rel_labels_batch = []
            for predictions in model_output:
                rel_labels = []
                rel_indices = np.nonzero(predictions)[0]
                for index in rel_indices:
                    if index == 0:
                        rel_labels.append("no relation")
                        continue
                    rel_p = self.id2rel[index]
                    rel_label = (rel_p, self.rel2label[rel_p])
                    rel_labels.append(rel_label)
                rel_labels_batch.append(rel_labels)
            return rel_labels_batch
        else:
            return model_output
