from pathlib import Path
import pandas as pd
from logging import exception, getLogger
from typing import List
import numpy as np

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


log = getLogger(__name__)

@register('dnnc_input_preprocessor')
class InputPreprocessor(Component):
    def __init__(self,
                 support_dataset_path: str,
                 format: str = "csv",
                 class_sep: str = ",",
                 *args, **kwargs) -> None:
        file = Path(support_dataset_path)
        if file.exists():
            if format == 'csv':
                keys = ('sep', 'header', 'names')
                options = {k: kwargs[k] for k in keys if k in kwargs}
                df = pd.read_csv(file, **options)
            elif format == 'json':
                keys = ('orient', 'lines')
                options = {k: kwargs[k] for k in keys if k in kwargs}
                df = pd.read_json(file, **options)
            else:
                raise Exception('Unsupported file format: {}'.format(format))

            x = kwargs.get("x", "text")
            y = kwargs.get('y', 'labels')
            
            multi_label = lambda labels: class_sep in str(labels)
            self.support_dataset = [[row[x], str(row[y])] for _, row in df.iterrows() if not multi_label(row[y])]
        else:
            log.warning("Cannot find {} file".format(support_dataset_path))
    
    def __call__(self,
                 input_texts : List[str]) -> List[List[str]]:
        # TODO: перепиши на numpy
        hypotesis_batch = []
        premise_batch = []
        hypotesis_labels_batch = []

        for [premise, [hypotesis, hypotesis_labels]] in zip(input_texts * len(self.support_dataset),
                                                            np.repeat(self.support_dataset, len(input_texts), axis=0)):
            premise_batch.append(premise)
            hypotesis_batch.append(hypotesis)
            hypotesis_labels_batch.append(hypotesis_labels)

        return hypotesis_batch, premise_batch, hypotesis_labels_batch



# @register('test42')
# class Test42(Component):
#     def __init__(self, *args, **kwargs) -> None:
#         pass

#     def __call__(self, input):
#         return input[:2]


# @register('dnnc_input_preprocessor')
# class PairMaker(Component): # TODO: а тут точно нужен component а не что-то ещё ?
#     def __init__(self, *args, **kwargs) -> None:
#         pass
#     def __call__(self, input_texts, support_texts, support_labels):
 
#         pairs = []
#         for x, y, z in zip(input_texts, support_texts, support_labels):
#             for i, j in zip(y, z):
#                 pairs.append([x, i, j])
#         pairs = np.array(pairs)
#         print()
#         print("PAIRS SHAPE = ", pairs.shape)
#         pairs = pairs.T.tolist()
#         transformed_input_texts = pairs[0]
#         transformed_input_texts = pairs[1]
#         transformed_support_labels = pairs[2]

#         print(f"PAIRS: transformed_inputs_batch = {transformed_input_texts[:3]}, len = {len(transformed_input_texts)}")
#         print(f"PAIRS: transformed_support_batch = {transformed_input_texts[:3]}, len = {len(transformed_input_texts)}")
#         print(f"PAIRS: transformed_support_labels = {transformed_support_labels[:3]}, len = {len(transformed_support_labels)}")
#         return transformed_input_texts, transformed_input_texts, transformed_support_labels

@register('dnnc_pair_maker')
class PairMaker(Component): # TODO: а тут точно нужен component а не что-то ещё ?
    def __init__(self, *args, **kwargs) -> None:
        pass
    def __call__(self, input_batch, support_batch):
 
        pairs = []
        for x, y in zip(input_batch, support_batch):
            for i in y:
                pairs.append([x, i])
        pairs = np.array(pairs)
        print()
        print("PAIRS SHAPE = ", pairs.shape)
        pairs = pairs.T.tolist()
        transformed_inputs_batch = pairs[0]
        transformed_support_batch = pairs[1]

        print(f"PAIRS: transformed_inputs_batch = {transformed_inputs_batch[:3]}, len = {len(transformed_inputs_batch)}")
        print(f"PAIRS: transformed_support_batch = {transformed_support_batch[:3]}, len = {len(transformed_support_batch)}")
        return transformed_inputs_batch, transformed_support_batch

@register('dnnc_printer')
class Printer(Component):
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def __call__(self, *args, **kwargs):
        print("________________PRINTER__________")
        print(**kwargs)
        print("_________________________________")

@register('dnnc_labels_to_binary')
class Labels2Binary(Component):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, true_labels, support_labels):
        return (true_labels == support_labels)

@register('dnnc_support_dataset_loader')
class SupportDatasetLoader(Component):
    def __init__(self,
                 path: str,
                 format: str = "csv",
                 class_sep: str = ",",
                 *args, **kwargs) -> None:
        file = Path(path)
        if file.exists():
            if format == 'csv':
                keys = ('sep', 'header', 'names')
                options = {k: kwargs[k] for k in keys if k in kwargs}
                df = pd.read_csv(file, **options)
            elif format == 'json':
                keys = ('orient', 'lines')
                options = {k: kwargs[k] for k in keys if k in kwargs}
                df = pd.read_json(file, **options)
            else:
                raise Exception('Unsupported file format: {}'.format(format))

            x = kwargs.get("x", "text")
            y = kwargs.get('y', 'labels')
            
            multi_label = lambda labels: class_sep in str(labels)
            support_dataset = [[row[x], str(row[y])] for _, row in df.iterrows() if not multi_label(row[y])]
            support_dataset = np.array(support_dataset)
            
            self.infer_support_texts = support_dataset[:, 0]
            self.infer_support_labels = support_dataset[:, 1]

        else:
            log.warning("Cannot find {} file".format(path))
    
    def __call__(self,
                 input_texts : List[str]) -> List[List[str]]:
        input_size = len(input_texts)

        print("input_size = ", input_size)
        print("self.infer_support_texts = ", self.infer_support_texts.size)
        print("self.infer_support_labels = ", self.infer_support_labels.size)
        return ([self.infer_support_texts] * input_size, [self.infer_support_labels] * input_size)


# @register('dnnc_support_dataset_loader')
# class SupportDatasetLoader(Component):
#     def __init__(self,
#                  path: str,
#                  format: str = "csv",
#                  class_sep: str = ",",
#                  *args, **kwargs) -> None:
#         file = Path(path)
#         if file.exists():
#             if format == 'csv':
#                 keys = ('sep', 'header', 'names')
#                 options = {k: kwargs[k] for k in keys if k in kwargs}
#                 df = pd.read_csv(file, **options)
#             elif format == 'json':
#                 keys = ('orient', 'lines')
#                 options = {k: kwargs[k] for k in keys if k in kwargs}
#                 df = pd.read_json(file, **options)
#             else:
#                 raise Exception('Unsupported file format: {}'.format(format))

#             x = kwargs.get("x", "text")
#             y = kwargs.get('y', 'labels')
            
#             multi_label = lambda labels: class_sep in str(labels)
#             self.support_dataset = [[row[x], str(row[y])] for _, row in df.iterrows() if not multi_label(row[y])]
                
#         else:
#             log.warning("Cannot find {} file".format(path))
    
#     def __call__(self, input_texts):
#         return np.array(self.support_dataset).T.tolist()