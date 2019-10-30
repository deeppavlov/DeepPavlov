# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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

import asyncio
import logging
import uuid
from functools import reduce
from pathlib import Path
from typing import Tuple, Optional, List

from rasa.cli.utils import get_validated_path
from rasa.constants import DEFAULT_MODELS_PATH
from rasa.core.agent import Agent
from rasa.core.channels import CollectingOutputChannel
from rasa.core.channels import UserMessage
from rasa.model import get_model

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

logger = logging.getLogger(__name__)


@register("rasa_skill")
class RASASkill(Component):
    """RASASkill lets you to wrap RASA Agent as a Skill within DeepPavlov environment.

    The component requires path to your RASA models (folder with timestamped tar.gz archieves)
    as you use in command `rasa run -m models --enable-api --log-file out.log`

    """

    def __init__(self, path_to_models: str, **kwargs) -> None:
        """
        Constructs RASA Agent as a DeepPavlov skill:
            read model folder,
            initialize rasa.core.agent.Agent and wrap it's interfaces

        Args:
            path_to_models: string path to folder with RASA models

        """
        # we need absolute path (expanded for user home and resolved if it relative path):
        self.path_to_models = Path(path_to_models).expanduser().resolve()

        model = get_validated_path(self.path_to_models, "model", DEFAULT_MODELS_PATH)

        model_path = get_model(model)
        if not model_path:
            # can not laod model path
            raise Exception("can not load model path: %s" % model)

        self._agent = Agent.load(model_path)
        self.ioloop = asyncio.new_event_loop()
        logger.info(f"path to RASA models is: `{self.path_to_models}`")

    def __call__(self,
                 utterances_batch: List[str],
                 states_batch: Optional[List] = None) -> Tuple[List[str], List[float], list]:
        """Returns skill inference result.

        Returns batches of skill inference results, estimated confidence
        levels and up to date states corresponding to incoming utterance
        batch.

        Args:
            utterances_batch: A batch of utterances of str type.
            states_batch:  A batch of arbitrary typed states for
                each utterance.


        Returns:
            response: A batch of arbitrary typed skill inference results.
            confidence: A batch of float typed confidence levels for each of
                skill inference result.
            output_states_batch:  A batch of arbitrary typed states for
                each utterance.

        """
        user_ids, output_states_batch = self._handle_user_identification(utterances_batch, states_batch)
        #################################################################################
        # RASA use asyncio for handling messages and handle_text is async function,
        # so we need to instantiate event loop
        # futures = [rasa_confident_response_decorator(self._agent, utt, sender_id=uid) for utt, uid in
        futures = [self.rasa_confident_response_decorator(self._agent, utt, sender_id=uid) for utt, uid in
                   zip(utterances_batch, user_ids)]

        asyncio.set_event_loop(self.ioloop)
        results = self.ioloop.run_until_complete(asyncio.gather(*futures))

        responses_batch, confidences_batch = zip(*results)
        return responses_batch, confidences_batch, output_states_batch

    async def rasa_confident_response_decorator(self, rasa_agent, text_message, sender_id):
        """
        Args:
            rasa_agent: rasa.core.agent.Agent instance
            text_message: str with utterance from user
            sender_id: id of the user

        Returns: None or tuple with str and float, where first element is a message and second is
            confidence
        """

        resp = await self.rasa_handle_text_verbosely(rasa_agent, text_message, sender_id)
        if resp:
            responses, confidences, actions = resp
        else:
            logger.warning("Null response from RASA Skill")
            return None

        # for adaptation to deep pavlov arch we need to merge multi-messages into single string:
        texts = [each_resp['text'] for each_resp in responses if 'text' in each_resp]
        merged_message = "\n".join(texts)

        merged_confidence = reduce(lambda a, b: a * b, confidences)
        # TODO possibly it better to choose another function for calculation of final confidence
        # current realisation of confidence propagation may cause confidence decay for long actions
        # chains. If long chains is your case, try max(confidence) or confidence[0]
        return merged_message, merged_confidence

    async def rasa_handle_text_verbosely(self, rasa_agent, text_message, sender_id):
        """
        This function reimplements RASA's rasa.core.agent.Agent.handle_text method to allow to retrieve
        message responses with confidence estimation altogether.

        It reconstructs with merge RASA's methods:
        https://github.com/RasaHQ/rasa_core/blob/master/rasa/core/agent.py#L401
        https://github.com/RasaHQ/rasa_core/blob/master/rasa/core/agent.py#L308
        https://github.com/RasaHQ/rasa/blob/master/rasa/core/processor.py#L327

        This required to allow RASA to output confidences with actions altogether
        (Out of the box RASA does not support such use case).

        Args:
            rasa_agent: rasa.core.agent.Agent instance
            text_message: str with utterance from user
            sender_id: id of the user

        Returns: None or
            tuple where first element is a list of messages dicts, the second element is a list
                of confidence scores for all actions (it is longer than messages list, because some actions
                does not produce messages)

        """
        message = UserMessage(text_message,
                              output_channel=None,
                              sender_id=sender_id)

        processor = rasa_agent.create_processor()
        tracker = processor._get_tracker(message.sender_id)

        confidences = []
        actions = []
        await processor._handle_message_with_tracker(message, tracker)
        # save tracker state to continue conversation from this state
        processor._save_tracker(tracker)

        # here we restore some of logic in RASA management.
        # ###### Loop of IntraStep decisions  ##########################################################
        # await processor._predict_and_execute_next_action(msg, tracker):
        # https://github.com/RasaHQ/rasa/blob/master/rasa/core/processor.py#L327-L362
        # keep taking actions decided by the policy until it chooses to 'listen'
        should_predict_another_action = True
        num_predicted_actions = 0

        def is_action_limit_reached():
            return (num_predicted_actions == processor.max_number_of_predictions and
                    should_predict_another_action)

        # action loop. predicts actions until we hit action listen
        while (should_predict_another_action and
               processor._should_handle_message(tracker) and
               num_predicted_actions < processor.max_number_of_predictions):
            # this actually just calls the policy's method by the same name
            action, policy, confidence = processor.predict_next_action(tracker)

            confidences.append(confidence)
            actions.append(action)

            should_predict_another_action = await processor._run_action(
                action,
                tracker,
                message.output_channel,
                processor.nlg,
                policy, confidence
            )
            num_predicted_actions += 1

        if is_action_limit_reached():
            # circuit breaker was tripped
            logger.warning(
                "Circuit breaker tripped. Stopped predicting "
                "more actions for sender '{}'".format(tracker.sender_id))
            if processor.on_circuit_break:
                # call a registered callback
                processor.on_circuit_break(tracker, message.output_channel, processor.nlg)

        if isinstance(message.output_channel, CollectingOutputChannel):

            return message.output_channel.messages, confidences, actions
        else:
            return None

    def _generate_user_id(self) -> str:
        """
        Here you put user id generative logic if you want to implement it in the skill.

        Although it is better to delegate user_id generation to Agent Layer
        Returns: str

        """
        return uuid.uuid1().hex

    def _handle_user_identification(self, utterances_batch, states_batch):
        """Method preprocesses states batch to guarantee that all users are identified (or
        identifiers are generated for all users).

        Args:
            utterances_batch: batch of utterances
            states_batch: batch of states

        Returns:

        """
        # grasp user_ids from states batch.
        # We expect that skill receives None or dict of state for each utterance.
        # if state has user_id then skill uses it, otherwise it generates user_id and calls the
        # user with this name in further.

        # In this implementation we use current datetime for generating uniqe ids
        output_states_batch = []
        user_ids = []
        if states_batch is None:
            # generate states batch matching batch of utterances:
            states_batch = [None] * len(utterances_batch)

        for state in states_batch:
            if not state:
                user_id = self._generate_user_id()
                new_state = {'user_id': user_id}

            elif 'user_id' not in state:
                new_state = state
                user_id = self._generate_user_id()
                new_state['user_id'] = self._generate_user_id()

            else:
                new_state = state
                user_id = new_state['user_id']

            user_ids.append(user_id)
            output_states_batch.append(new_state)
        return user_ids, output_states_batch

    def destroy(self):
        self.ioloop.close()
        super().destroy()
