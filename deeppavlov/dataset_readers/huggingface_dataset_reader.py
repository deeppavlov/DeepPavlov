# Copyright 2020 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import re
from collections import Counter
from math import floor
from typing import Dict, Optional, List, Union

from datasets import load_dataset, Dataset, Features, ClassLabel
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('huggingface_dataset_reader')
class HuggingFaceDatasetReader(DatasetReader):
    """Adds HuggingFace Datasets https://huggingface.co/datasets/ to DeepPavlov
    """

    @overrides
    def read(self,
             data_path: str,
             path: str,
             name: Optional[str] = None,
             train: str = 'train',
             valid: Optional[str] = None,
             test: Optional[str] = None,
             **kwargs) -> Dict[str, Dataset]:
        """Wraps datasets.load_dataset method

        Args:
            data_path: DeepPavlov's data_path argument, is not used, but passed by trainer
            path: datasets.load_dataset path argument (e.g., `glue`)
            name: datasets.load_dataset name argument (e.g., `mrpc`)
            train: split name to use as training data.
            valid: split name to use as validation data.
            test: split name to use as test data.

        Returns:
            Dict[str, List[Dict]]: Dictionary with train, valid, test datasets
        """
        if 'split' in kwargs:
            raise RuntimeError('Split argument was used. Use train, valid, test arguments instead of split.')

        # pop elements not relevant to BuilderConfig
        downsample_ratio: Union[List[float], float] = kwargs.pop("downsample_ratio", 1.)
        seed = kwargs.pop("seed", 42)
        percentage = kwargs.pop("dev_percentage", 50)
        do_index_correction = kwargs.pop("do_index_correction", True)

        split_mapping = {'train': train, 'valid': valid, 'test': test}
        # filter unused splits
        split_mapping = {el: split_mapping[el] for el in split_mapping if split_mapping[el]}

        if isinstance(downsample_ratio, float):
            downsample_ratio = [downsample_ratio] * len(split_mapping)
        elif isinstance(downsample_ratio, list) and len(downsample_ratio) != len(split_mapping):
            raise ValueError("The number of downsample ratios must be the same as the number of splits")

        dataset = load_dataset(path=path, name=name, split=list(split_mapping.values()), **kwargs)
        if path == "super_glue" and name == "copa":
            dataset = [dataset_split.map(preprocess_copa, batched=True) for dataset_split in dataset]
        elif path == "super_glue" and name == "boolq":
            dataset = load_dataset(path=path,
                                   name=name,
                                   split=interleave_splits(splits=list(split_mapping.values()),
                                                           percentage=percentage),
                                   **kwargs)
            dataset = [dataset_split.map(preprocess_boolq, batched=True) for dataset_split in dataset]
        elif path == "super_glue" and name == "record":
            label_column = "label"
            dataset = [
                binary_downsample(
                    add_label_names(
                        dataset_split.map(preprocess_record,
                                          batched=True,
                                          remove_columns=["answers"]),
                        label_column=label_column,
                        label_names=["False", "True"]
                    ),
                    ratio=ratio,
                    seed=seed,
                    label_column=label_column,
                    do_correction=do_index_correction
                ).map(add_num_examples, batched=True, batch_size=None)
                for dataset_split, ratio
                in zip(dataset, downsample_ratio)
            ]
        return dict(zip(split_mapping.keys(), dataset))


def interleave_splits(splits: List[str], percentage: int = 50) -> List[str]:
    """Adds a portion of `dev` (or, `test` if there's only `train` and `test`) set to the `train` set.
    Assumes that there are at two splits are passed ordered as (train, dev, test).
    Args:
        splits: list of strings
        percentage: percentage (represented as an integer value between 0 and 100)
                    of samples to extract from `dev` and add to `train`
    Returns:
        List[str] containing mixing instructions (e.g. ['train+validation[:50%]', 'validation[-50%:]'])
    """
    if len(splits) < 2:
        raise ValueError("At least two splits should be passed to this function")
    mixed_splits = [f"{splits[0]}+{splits[1]}[:{percentage}%]", f"{splits[1]}[-{percentage}%:]"]
    if len(splits) == 3:
        mixed_splits += [splits[2]]
    return mixed_splits


def preprocess_copa(examples: Dataset) -> Dict[str, List[List[str]]]:
    """COPA preprocessing to be applied by the map function.
    Args:
        examples: an instance of Dataset class
    Returns:
        Dict[str, List[List[str]]]: processed features represented as nested
        list with number of elements corresponding to the number of choices
        (2 in this case)
    """
    question_dict = {
        "cause": "What was the cause of this?",
        "effect": "What happened as a result?",
    }

    num_choices = 2

    questions = [question_dict[question] for question in examples["question"]]
    premises = examples["premise"]

    contexts = [f"{premise} {question}" for premise, question in zip(premises, questions)]
    contexts = [[context] * num_choices for context in contexts]

    choices = [[choice1, choice2] for choice1, choice2 in zip(examples["choice1"], examples["choice2"])]

    return {"contexts": contexts,
            "choices": choices}


def preprocess_boolq(examples: Dataset) -> Dict[str, List[str]]:
    """BoolQ preprocessing to be applied by the map function. The preprocessing boils down
    to removing redundant titles from the passages.
    Args:
        examples: an instance of Dataset class
    Returns:
        Dict[str, List[str]]: processed features (just the passage in this case)
    """

    def remove_passage_title(passage: str) -> str:
        """Removes the title of a given passage. The motivation is that the title duplicates
        the beginning of the text body, which means that it's redundant. We remove to save space.
        Args:
            passage: a single `passage` string
        Returns:
            str: the same `passage` string with the title removed
        """
        return re.sub(r"^.+-- ", "", passage)

    passages = [remove_passage_title(passage) for passage in examples["passage"]]

    return {"passage": passages}


def preprocess_record(examples: Dataset) -> Dict[str,
                                                 Union[List[str],
                                                       List[int]]]:
    """ReCoRD preprocessing to be applied by the map function. This transforms the original
    nested structure of the dataset into a flat one. New indices are generated to allow for
    the restoration of the original structure. The resulting dataset amounts to a binary
    classification problem.
    Args:
        examples: an instance of Dataset class
    Returns:
        Dict[str, Union[List[str], List[int]]]: flattened features of the dataset
    """

    def fill_placeholder(sentence: str, candidate: str) -> str:
        """Fills `@placeholder` of a given query with the provided entity
        Args:
            sentence: query to fill
            candidate: entity candidate for the query
        Returns:
            str: `sentence` with `@placeholder` replaced with `candidate`
        """
        return re.sub(r"@placeholder", candidate.replace("\\", ""), sentence)

    def remove_highlight(context: str) -> str:
        """Removes highlights from a given passage
        Args:
            context: a passage to remove highlights from
        Returns:
            str: `context` with highlights removed
        """
        return re.sub(r"\n@highlight\n", ". ", context)

    queries: List[str] = examples["query"]
    passages: List[str] = [remove_highlight(passage) for passage in examples["passage"]]
    answers: List[List[str]] = examples["answers"]
    entities: List[List[str]] = examples["entities"]
    indices: List[Dict[str, int]] = examples["idx"]

    # new indices for flat examples
    merged_indices: List[str] = []
    # queries with placeholders filled
    filled_queries: List[str] = []
    # duplicated passages
    extended_passages: List[str] = []
    # contains one entity per flat example
    flat_entities: List[str] = []
    # whether the entity in this example is found in the answers (0 or 1)
    labels: List[int] = []

    for query, passage, list_of_answers, list_of_entities, index in zip(queries,
                                                                        passages,
                                                                        answers,
                                                                        entities,
                                                                        indices):
        num_candidates: int = len(list_of_entities)

        candidate_queries: List[str] = [fill_placeholder(query, entity) for entity in list_of_entities]
        cur_labels: List[int] = [int(entity in list_of_answers) if list_of_answers else -1 for entity in
                                 list_of_entities]
        cur_passages: List[str] = [passage] * num_candidates

        # keep track of the indices to be able to use target metrics
        passage_index: int = index["passage"]
        query_index: int = index["query"]
        example_indices: List[str] = [f"{passage_index}-{query_index}-{num_candidates}"] * num_candidates

        merged_indices.extend(example_indices)
        filled_queries.extend(candidate_queries)
        extended_passages.extend(cur_passages)
        flat_entities.extend(list_of_entities)
        labels.extend(cur_labels)

    return {"idx": merged_indices,
            "query": filled_queries,
            "passage": extended_passages,
            "entities": flat_entities,
            "label": labels}


def add_label_names(dataset: Dataset, label_column: str, label_names: List[str]):
    """Adds `names` to a specified `label` column.
    All labels (i.e. integers) in the dataset should be < than the number of label names.
    Args:
        dataset: a Dataset to add label names to
        label_column: the name of the label column (such as `label` or `labels`) in the dataset
        label_names: a list of label names
    Returns:
        Dataset: A copy of the passed `dataset` with added label names
    """
    new_features: Features = dataset.features.copy()
    new_features[label_column] = ClassLabel(names=label_names)
    return dataset.cast(new_features)


def binary_downsample(dataset: Dataset,
                      ratio: float = 0.,
                      seed: int = 42,
                      label_column: str = "label",
                      *,
                      do_correction: bool = True) -> Dataset:
    """Downsamples a given dataset to the specified negative to positive examples ratio. Only works with
    binary classification datasets with labels denoted as `0` and `1`.
    Args:
        dataset: a Dataset to downsample
        ratio: negative to positive examples ratio to maintain
        seed: a seed for shuffling
        label_column: the name of `label` column such as 'label' or 'labels'
        do_correction: correct resampled indices. If indices aren't corrected then examples with mismatched
        indices will not be accounted for be ReCoRD metrics. This is not necessarily undesirable because
        examples with such indices will have less negative examples (or even none), which makes them easier
        for the model, thus inflating the resulting metrics.
    Returns:
        Dataset: a downsampled dataset
    """

    def replace_indices(data: Dataset, index_map: Dict[str, str]) -> Dict[str, List[str]]:
        idx: List[str] = [index_map.get(el, el) for el in data["idx"]]
        return {"idx": idx}

    def get_correct_indices_map(data: Dataset) -> Dict[str, str]:
        """Generate a dictionary with replacements for indices that
        are no longer correct due to downsampling (i.e. the total number
        of elements denoted by the last part of an index has changed)
        Args:
            data: a downsampled Dataset
        Returns:
            Dict[str, str]: a dictionary containing replacement indices
        """
        actual_n_elements: Counter = Counter(data["idx"])
        corrected_index_map: Dict[str, str] = dict()
        for idx, n_elements in actual_n_elements.items():
            expected_n_elements: int = int(idx.split("-")[-1])
            if expected_n_elements != n_elements:
                new_idx: List[str] = idx.split("-")
                new_idx[-1]: str = str(n_elements)
                new_idx: str = "-".join(new_idx)
                corrected_index_map[idx] = new_idx
        return corrected_index_map

    def correct_indices(data: Dataset) -> Dataset:
        """Sets correct number of examples in downsampled indices
        Args:
            data: a downsampled dataset
        Returns:
            Dataset: the same dataset with correct indices
        """
        index_map: Dict[str, str] = get_correct_indices_map(data)
        return data.map(replace_indices, batched=True, fn_kwargs={"index_map": index_map})

    dataset_labels = dataset.unique(label_column)
    # `test` split shouldn't be downsampled
    if dataset_labels == [-1]:
        return dataset
    elif dataset_labels == [0, 1]:
        # positive examples are denoted with `1`
        num_positive: int = sum(dataset[label_column])
        num_total: int = len(dataset)
        # the original number of negative examples is returned if `ratio` is not explicitly specified
        num_negative: int = floor(num_positive * ratio if ratio > 0 else num_total - num_positive)
        # first `num_positive` examples in a sorted dataset are labeled with `1`
        # while the rest are labeled with `0`
        sorted_dataset: Dataset = dataset.sort(label_column, reverse=True)
        # but we need to reshuffle the dataset before returning it
        shuffled_dataset: Dataset = sorted_dataset.select(range(num_positive + num_negative)).shuffle(seed=seed)
        if do_correction:
            shuffled_dataset = correct_indices(shuffled_dataset)
        return shuffled_dataset
    # the same logic is not applicable to cases with != 2 classes
    else:
        raise ValueError("Only binary classification labels are supported (i.e. [0, 1])")


def add_num_examples(dataset: Dataset) -> Dict[str, List[int]]:
    """Adds the total number of examples in a given dataset to
    each individual example. Must be applied to the whole dataset (i.e. `batched=True, batch_size=None`),
    otherwise the number will be incorrect.
    Args:
        dataset: a Dataset to add number of examples to
    Returns:
        Dict[str, List[int]]: total number of examples repeated for each example
    """
    num_examples = len(dataset[next(iter(dataset))])
    return {"num_examples": [num_examples] * num_examples}
