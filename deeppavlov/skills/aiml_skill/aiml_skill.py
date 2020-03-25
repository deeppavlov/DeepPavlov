# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

import uuid
from logging import getLogger
from pathlib import Path
from typing import Tuple, Optional, List

import aiml

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register("aiml_skill")
class AIMLSkill(Component):
    """Skill wraps python-aiml library into DeepPavlov interfrace.
    AIML uses directory with AIML scripts which are loaded at initialization and used as patterns
    for answering at each step.
    """

    def __init__(self,
                 path_to_aiml_scripts: str,
                 positive_confidence: float = 0.66,
                 null_response: str = "I don't know what to answer you",
                 null_confidence: float = 0.33,
                 **kwargs
                 ) -> None:
        """
        Construct skill:
            read AIML scripts,
            load AIML kernel

        Args:
            path_to_aiml_scripts: string path to folder with AIML scripts
            null_response: Response string to answer if no AIML Patterns matched
            positive_confidence: The confidence of response if response was found in AIML scripts
            null_confidence: The confidence when AIML scripts has no rule for responding and system returns null_response
        """
        # we need absolute path (expanded for user home and resolved if it relative path):
        self.path_to_aiml_scripts = Path(path_to_aiml_scripts).expanduser().resolve()
        log.info(f"path_to_aiml_scripts is: `{self.path_to_aiml_scripts}`")

        self.positive_confidence = positive_confidence
        self.null_confidence = null_confidence
        self.null_response = null_response
        self.kernel = aiml.Kernel()
        # to block AIML output:
        self.kernel._verboseMode = False
        self._load_scripts()

    def _load_scripts(self) -> None:
        """
        Scripts are loaded recursively from files with extensions .xml and .aiml
        Returns: None

        """
        # learn kernel to all aimls in directory tree:
        all_files = sorted(self.path_to_aiml_scripts.rglob('*.*'))
        learned_files = []
        for each_file_path in all_files:
            if each_file_path.suffix in ['.aiml', '.xml']:
                # learn the script file
                self.kernel.learn(str(each_file_path))
                learned_files.append(each_file_path)
        if not learned_files:
            log.warning(f"No .aiml or .xml files found for AIML Kernel in directory {self.path_to_aiml_scripts}")

    def process_step(self, utterance_str: str, user_id: any) -> Tuple[str, float]:
        response = self.kernel.respond(utterance_str, sessionID=user_id)
        # here put your estimation of confidence:
        if response:
            # print(f"AIML responds: {response}")
            confidence = self.positive_confidence
        else:
            # print("AIML responses silently...")
            response = self.null_response
            confidence = self.null_confidence
        return response, confidence

    def _generate_user_id(self) -> str:
        """Here you put user id generative logic if you want to implement it in the skill.

        Returns:
            user_id: Random generated user ID.

        """
        return uuid.uuid1().hex

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

        confident_responses = map(self.process_step, utterances_batch, user_ids)
        responses_batch, confidences_batch = zip(*confident_responses)

        return responses_batch, confidences_batch, output_states_batch
