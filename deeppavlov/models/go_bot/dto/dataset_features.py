from typing import List

import numpy as np


# todo remove boilerplate duplications
# todo comments
# todo logging
# todo naming
from deeppavlov.models.go_bot.nlu.dto.nlu_response import NLUResponse
from deeppavlov.models.go_bot.policy.dto.digitized_policy_features import DigitizedPolicyFeatures
from deeppavlov.models.go_bot.tracker.dto.dst_knowledge import DSTKnowledge

from copy import deepcopy


class UtteranceFeatures:
    """
    the DTO-like class storing the training features of a single utterance of a dialog
    (to feed the GO-bot policy model)
    """

    action_mask: np.ndarray
    attn_key: np.ndarray
    tokens_embeddings_padded: np.ndarray
    features: np.ndarray

    def __init__(self,
                 nlu_response: NLUResponse,
                 tracker_knowledge: DSTKnowledge,
                 features: DigitizedPolicyFeatures):
        self.action_mask = features.action_mask
        self.attn_key = features.attn_key

        tokens_vectorized = nlu_response.tokens_vectorized  # todo proper oop
        self.tokens_embeddings_padded = tokens_vectorized.tokens_embeddings_padded
        self.features = features.concat_feats


class UtteranceTarget:
    """
    the DTO-like class storing the training target of a single utterance of a dialog
    (to feed the GO-bot policy model)
    """
    action_id: int

    def __init__(self, action_id):
        self.action_id = action_id


class UtteranceDataEntry:
    """
    the DTO-like class storing both the training features and target
    of a single utterance of a dialog (to feed the GO-bot policy model)
    """
    features: UtteranceFeatures
    target: UtteranceTarget

    def __init__(self, features, target):
        self.features = features
        self.target = target

    @staticmethod
    def from_features_and_target(features: UtteranceFeatures, target: UtteranceTarget):
        return UtteranceDataEntry(deepcopy(features), deepcopy(target))

    @staticmethod
    def from_features(features: UtteranceFeatures):
        return UtteranceDataEntry(deepcopy(features), UtteranceTarget(None))


class DialogueFeatures:
    """
    the DTO-like class storing both the training features
    of a dialog (to feed the GO-bot policy model)
    """
    action_masks: List[np.ndarray]
    attn_keys: List[np.ndarray]
    tokens_embeddings_paddeds: List[np.ndarray]
    featuress: List[np.ndarray]

    def __init__(self):
        self.action_masks = []
        self.attn_keys = []
        self.tokens_embeddings_paddeds = []
        self.featuress = []

    def append(self, utterance_features: UtteranceFeatures):
        self.action_masks.append(utterance_features.action_mask)
        self.attn_keys.append(utterance_features.attn_key)
        self.tokens_embeddings_paddeds.append(utterance_features.tokens_embeddings_padded)
        self.featuress.append(utterance_features.features)

    def __len__(self):
        return len(self.featuress)


class DialogueTargets:
    """
    the DTO-like class storing both the training targets
    of a dialog (to feed the GO-bot policy model)
    """
    action_ids: List[int]

    def __init__(self):
        self.action_ids = []

    def append(self, utterance_target: UtteranceTarget):
        self.action_ids.append(utterance_target.action_id)

    def __len__(self):
        return len(self.action_ids)


class DialogueDataEntry:
    """
    the DTO-like class storing both the training features and targets
    of a dialog (to feed the GO-bot policy model)
    """
    features: DialogueFeatures
    targets: DialogueTargets

    def __init__(self):
        self.features = DialogueFeatures()
        self.targets = DialogueTargets()

    def append(self, utterance_features: UtteranceDataEntry):
        self.features.append(utterance_features.features)
        self.targets.append(utterance_features.target)

    def __len__(self):
        return len(self.features)


class PaddedDialogueFeatures(DialogueFeatures):
    """
    the DTO-like class storing both the **padded to some specified length** training features
    of a dialog (to feed the GO-bot policy model)
    """
    padded_dialogue_length_mask: List[int]

    def __init__(self, dialogue_features: DialogueFeatures, sequence_length):
        super().__init__()

        padding_length = sequence_length - len(dialogue_features)

        self.padded_dialogue_length_mask = [1] * len(dialogue_features) + [0] * padding_length

        self.action_masks = dialogue_features.action_masks + \
                            [np.zeros_like(dialogue_features.action_masks[0])] * padding_length

        self.attn_keys = dialogue_features.attn_keys + [np.zeros_like(dialogue_features.attn_keys[0])] * padding_length

        self.tokens_embeddings_paddeds = dialogue_features.tokens_embeddings_paddeds + \
                                         [np.zeros_like(
                                             dialogue_features.tokens_embeddings_paddeds[0])] * padding_length

        self.featuress = dialogue_features.featuress + [np.zeros_like(dialogue_features.featuress[0])] * padding_length


class PaddedDialogueTargets(DialogueTargets):
    """
    the DTO-like class storing both the **padded to some specified length** training targets
    of a dialog (to feed the GO-bot policy model)
    """
    def __init__(self, dialogue_targets: DialogueTargets, sequence_length):
        super().__init__()

        padding_length = sequence_length - len(dialogue_targets)
        self.action_ids = dialogue_targets.action_ids + [0] * padding_length


class PaddedDialogueDataEntry(DialogueDataEntry):
    """
    the DTO-like class storing both the **padded to some specified length** training features and targets
    of a dialog (to feed the GO-bot policy model)
    """
    features: PaddedDialogueFeatures
    targets: PaddedDialogueTargets

    def __init__(self, dialogue_data_entry: DialogueDataEntry, sequence_length):
        super().__init__()

        self.features = PaddedDialogueFeatures(dialogue_data_entry.features, sequence_length)
        self.targets = PaddedDialogueTargets(dialogue_data_entry.targets, sequence_length)


class BatchDialoguesFeatures:
    """
    the DTO-like class storing both the training features
    of a batch of dialogues. (to feed the GO-bot policy model)
    """
    b_action_masks: List[List[np.ndarray]]
    b_attn_keys: List[List[np.ndarray]]
    b_tokens_embeddings_paddeds: List[List[np.ndarray]]
    b_featuress: List[List[np.ndarray]]
    b_padded_dialogue_length_mask: List[List[int]]
    max_dialogue_length: int

    def __init__(self, max_dialogue_length):
        self.b_action_masks = []
        self.b_attn_keys = []
        self.b_tokens_embeddings_paddeds = []
        self.b_featuress = []
        self.b_padded_dialogue_length_mask = []
        self.max_dialogue_length = max_dialogue_length

    def append(self, padded_dialogue_features: PaddedDialogueFeatures):
        self.b_action_masks.append(padded_dialogue_features.action_masks)
        self.b_attn_keys.append(padded_dialogue_features.attn_keys)
        self.b_tokens_embeddings_paddeds.append(padded_dialogue_features.tokens_embeddings_paddeds)
        self.b_featuress.append(padded_dialogue_features.featuress)
        self.b_padded_dialogue_length_mask.append(padded_dialogue_features.padded_dialogue_length_mask)

    def __len__(self):
        return len(self.b_featuress)


class BatchDialoguesTargets:
    """
    the DTO-like class storing both the training targets
    of a batch of dialogues. (to feed the GO-bot policy model)
    """
    b_action_ids: List[List[int]]
    max_dialogue_length: int

    def __init__(self, max_dialogue_length):
        self.b_action_ids = []
        self.max_dialogue_length = max_dialogue_length

    def append(self, padded_dialogue_targets: PaddedDialogueTargets):
        self.b_action_ids.append(padded_dialogue_targets.action_ids)

    def __len__(self):
        return len(self.b_action_ids)


class BatchDialoguesDataset:
    """
    the DTO-like class storing both the training features and target
    of a batch of dialogues. (to feed the GO-bot policy model)
    Handles the dialogues padding.
    """
    features: BatchDialoguesFeatures
    targets: BatchDialoguesTargets

    def __init__(self, max_dialogue_length):
        self.features = BatchDialoguesFeatures(max_dialogue_length)
        self.targets = BatchDialoguesTargets(max_dialogue_length)
        self.max_dialogue_length = max_dialogue_length

    def append(self, dialogue_features: DialogueDataEntry):
        padded_dialogue_features = PaddedDialogueDataEntry(dialogue_features, self.max_dialogue_length)
        self.features.append(padded_dialogue_features.features)
        self.targets.append(padded_dialogue_features.targets)

    def __len__(self):
        return len(self.features)
