import sys
from typing import Type
import numpy as np

from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.common.registry import register_model
from deeppavlov.core.data.utils import load_vocab
from deeppavlov.core.models.inferable import Inferable

from deeppavlov.models.hcn.models.at import ActionTracker
from deeppavlov.models.hcn.models.bow import BoW_encoder
from deeppavlov.models.hcn.models.embedder import UtteranceEmbed
from deeppavlov.models.hcn.models.et import EntityTracker
from deeppavlov.models.hcn.models.lstm import LSTM


@register_model("hcn_go")
class HybridCodeNetwork(Inferable, Trainable):
    def __init__(self, vocab_path, bow_encoder: Type = BoW_encoder, embedder: Type = UtteranceEmbed,
                 entity_tracker: Type = EntityTracker):

        self.bow_encoder = bow_encoder
        self.embedder = embedder
        self.entity_tracker = entity_tracker
        self.action_tracker = ActionTracker(self.entity_tracker)

        self.vocab = load_vocab(vocab_path)
        input_size = self.embedder.dim + len(self.vocab) + self.entity_tracker.num_features
        self.net = LSTM(input_size=input_size, output_size=self.action_tracker.action_size)

    def train(self, data, num_epochs, num_tr_dialogs, acc_threshold=0.99):

        # TODO `data` should be batch

        print('\n:: training started\n')

        tr_data = data[:num_tr_dialogs]
        eval_data = data[num_tr_dialogs:250]
        # num_tr_instances = sum(len(dialog) for dialog in tr_data)
        for j in range(num_epochs):
            loss = 0.
            i = 0
            for dialog in tr_data:

                self.reset()

                dialog_loss = 0.
                for pair in dialog:
                    # Loss for a single context-response pair
                    dialog_loss += self.net.train(self._encode_context(pair['context']),
                                                  self._encode_response(pair['response']),
                                                  self.action_tracker.action_mask())
                # A whole dialog(batch) loss
                dialog_loss /= len(dialog)

                # Epoch loss
                loss += dialog_loss
                i += 1
                sys.stdout.write('\r{}.[{}/{}]'.format(j + 1, i, len(tr_data)))

            print('\n\n:: {}.tr loss {}'.format(j + 1, loss / len(tr_data)))

            accuracy = self.evaluate(eval_data)
            print(':: {}.dev accuracy {}\n'.format(j + 1, accuracy))

            if accuracy > acc_threshold:
                print('Accuracy is {}, training stopped.'.format(accuracy))
                self.net.save()
                break
        self.net.save()

    def evaluate(self, eval_data):
        num_eval_dialogs = len(eval_data)
        dialog_accuracy = 0.

        for dialog in eval_data:

            self.reset()

            # iterate through dialog
            correct_examples = 0
            for pair in dialog:
                features = self._encode_context(pair['context'])
                # get action mask
                action_mask = self.action_tracker.action_mask()
                # forward propagation
                #  train step
                prediction = self.net.infer(features, action_mask)
                correct_examples += int(prediction == self._encode_response(pair['response']))
            # get dialog accuracy
            dialog_accuracy += correct_examples / len(dialog)

        return dialog_accuracy / num_eval_dialogs

    def _encode_context(self, context):
        """
        Gen feature for train step.
        :return: training input vector
        """
        context = context.lower()
        bow = self.bow_encoder.infer(context, self.vocab)
        emb = self.embedder.infer(context)
        self.entity_tracker.infer(context)
        entities = self.entity_tracker.context_features()
        features = np.concatenate((entities, emb, bow), axis=0)
        return features

    def _encode_response(self, response):
        return self.action_tracker.get_template_id(response)

    def infer(self, context):
        if context == 'clear' or context == 'reset' or context == 'restart':
            self.reset()
            return ''
        if not context:
            context = '<SILENCE>'
        features = self._encode_context(context)
        action_mask = self.action_tracker.action_mask()
        pred = self.net.infer(features, action_mask)
        return self.action_tracker.action_templates[pred]

    def reset(self):
        self.entity_tracker.reset()
        self.action_tracker.reset(self.entity_tracker)
        self.net.reset_state()
