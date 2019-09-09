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

import json
from collections import defaultdict
from typing import Callable, Optional, DefaultDict, Any

from deeppavlov import train_model
from deeppavlov.skills.dsl_skill.context import UserContext
from deeppavlov.skills.dsl_skill.utils import execute_query
from .handler import Handler


class FAQStorage:
    """
    Helper class, that works with database.

    Args
        handler_name: name of handler function

    """

    def __init__(self, handler_name: str, db_name: str = "faq.db"):
        self.handler_name = handler_name
        self.connection_string = db_name
        execute_query(self.connection_string, "CREATE TABLE IF NOT EXISTS handler_intent("
                                              "handler_name VARCHAR(100) NOT NULL,"
                                              " intent_name VARCHAR(100) NOT NULL,"
                                              " UNIQUE(intent_name, intent_name))")
        execute_query(self.connection_string, "CREATE TABLE IF NOT EXISTS intent_phrase("
                                              "intent_name VARCHAR(100) NOT NULL,"
                                              " phrase VARCHAR(500) NOT NULL,"
                                              " UNIQUE(intent_name, phrase))")
        execute_query(self.connection_string, "CREATE TABLE IF NOT EXISTS intent_answer("
                                              "intent_name VARCHAR(100) PRIMARY KEY,"
                                              " answer VARCHAR(500) NOT NULL)")
        execute_query(self.connection_string, "CREATE TABLE IF NOT EXISTS intent_metadata("
                                              "intent_name VARCHAR(100) PRIMARY KEY,"
                                              " metadata VARCHAR(500) NOT NULL)")

    def get_handler_intents(self):
        """
        Returns handler intents

        Returns
            handler intents

        """
        sql_query = "SELECT intent_name FROM handler_intent WHERE handler_name = (?)"
        result = execute_query(self.connection_string, sql_query, (self.handler_name,))
        data = result.fetchall()
        intent_names = [intent[0] for intent in data]
        return intent_names

    def set_faq_dict(self, faq_dict: dict):
        """
        Adds intents from faq_dict to the database

        Args
         faq_dict: FAQ dictionary

        """
        for intent_name, intent_dict in faq_dict.items():
            self.add_intent(intent_name, intent_dict)

    def get_faq_dict(self) -> DefaultDict[Any, DefaultDict[Any, list]]:
        """
        Returns FAQ dictionary from the database

        Returns
            FAQ dictionary

        """
        faq_dict = defaultdict(lambda: defaultdict(lambda: []))

        # phrases
        sql_query = "SELECT * FROM intent_phrase"
        result = execute_query(self.connection_string, sql_query)
        rows = result.fetchall()
        for intent_name, phrase in rows:
            faq_dict[intent_name]['phrases'].append(phrase)

        # answer
        sql_query = "SELECT * FROM intent_answer"
        result = execute_query(self.connection_string, sql_query)
        rows = result.fetchall()
        for intent_name, answer in rows:
            faq_dict[intent_name]['answer'] = answer

        # metadata
        sql_query = "SELECT * FROM intent_metadata"
        result = execute_query(self.connection_string, sql_query)
        rows = result.fetchall()
        for intent_name, metadata in rows:
            faq_dict[intent_name]['metadata'] = json.loads(metadata)

        return faq_dict

    def add_intent(self, intent_name: str, intent_dict: dict):
        """
        Adds intent to the database

        Args
            intent_name: intent name
            intent_dict: intent dictionary

        """
        # handler
        sql_query = "INSERT INTO handler_intent(handler_name, intent_name) VALUES (?, ?);"
        execute_query(self.connection_string, sql_query, (self.handler_name, intent_name))
        # phrases
        for phrase in intent_dict['phrases']:
            self.add_phrase(intent_name, phrase)
        # answer
        sql_query = "INSERT INTO intent_answer(intent_name, answer) VALUES (?, ?);"
        execute_query(self.connection_string, sql_query, (intent_name, intent_dict['answer']))
        # metadata
        sql_query = "INSERT INTO intent_metadata(intent_name, metadata) VALUES (?, ?);"
        execute_query(self.connection_string, sql_query, (intent_name, json.dumps(intent_dict['metadata'])))

    def remove_intent(self, intent_name: str):
        """
        Removes intent from the database

        Args
            intent_name: intent name

        """
        # phrases
        sql_query = "DELETE FROM intent_phrase WHERE intent_name = (?);"
        execute_query(self.connection_string, sql_query, (intent_name,))
        # answer
        sql_query = "DELETE FROM intent_answer WHERE intent_name = (?);"
        execute_query(self.connection_string, sql_query, (intent_name,))
        # metadata
        sql_query = "DELETE FROM intent_metadata WHERE intent_name = (?);"
        execute_query(self.connection_string, sql_query, (intent_name,))

    def add_phrase(self, intent_name: str, phrase: str):
        """
        Adds `phrase` to intent dictionary

        Args
            intent_name: intent name
            phrase: phrase
        """
        sql_query = "INSERT INTO intent_phrase(intent_name, phrase) VALUES (?, ?);"
        execute_query(self.connection_string, sql_query, (intent_name, phrase))

    def remove_phrase(self, intent_name: str, phrase: str):
        """
        Removes `phrase` from intent dictionary

        Args
            intent_name: intent name
            phrase: phrase
        """
        sql_query = "DELETE FROM intent_phrase WHERE intent_name = (?) AND phrase = (?);"
        execute_query(self.connection_string, sql_query, (intent_name, phrase))

    def change_answer(self, intent_name: str, answer: str):
        """
        Changes answer in intent dictionary

        Args
            intent_name: intent name
            answer: answer

        """
        sql_query = "UPDATE intent_answer SET(intent_name, phrase) = (?, ?) WHERE intent_name = (?);"
        execute_query(self.connection_string, sql_query, (intent_name, answer, intent_name))


class FAQHandler(Handler):
    """
    FAQHandler performs matching between the user request and FAQ database using a specified matching model.

    Attributes:
        func: handler function
        model_config: DeepPavlov-compatible model config for an FAQ skill
        score_threshold: [0-1], a lower bound on the acceptable closeness of the user question to
         a question in the FAQ
        state: state in which handler can be activated
        priority: priority of the function. If 2 or more handlers can be activated, function
         with the highest priority is selected
        context_condition: predicate that accepts user context and checks if the handler should be activated.
         Example: `lambda context: context.user_id != 1` checks if user_id is not equal to 1.
         That means a user with id 1 will be always ignored by the handler.
        storage: faq dictionary storage

    """

    def __init__(self,
                 func: Callable,
                 model_config: dict,
                 score_threshold: float,
                 state: Optional[str] = None,
                 context_condition: Optional[Callable] = None,
                 priority: int = 0):
        super().__init__(func, state, context_condition, priority)
        self.model_config = model_config
        self.score_threshold = score_threshold
        self.faq_model = None
        self.storage = FAQStorage(func.__name__)

    def train(self):
        """
        Trains the model
        """
        self.model_config['dataset_reader']['class_name'] = "faq_dict_reader"
        self.model_config['dataset_reader']['data'] = self.get_faq_dict()
        self.faq_model = train_model(self.model_config, download=True)
        return self

    def set_faq_dict(self, faq_dict):
        """
        Adds intents from faq_dict to the database and then trains the model

        Args:
            faq_dict: FAQ dictionary
        """
        self.storage.set_faq_dict(faq_dict)
        self.train()

    def add_faq_dict(self, faq_dict):
        """
        Adds intents from faq_dict to the database. Does not train model!

        Args:
            faq_dict: FAQ dictionary
        """
        self.storage.set_faq_dict(faq_dict)

    def get_faq_dict(self) -> DefaultDict[Any, DefaultDict[Any, list]]:
        """
        Returns FAQ dictionary from the database

        Returns:
            FAQ dictionary
        """
        return self.storage.get_faq_dict()

    def add_intent(self, intent_name: str, intent_dict: dict):
        """
        Adds intent to the database and then trains the model

        Args:
            intent_name: intent name
            intent_dict: intent dictionary

        """
        self.storage.add_intent(intent_name, intent_dict)
        self.train()

    def remove_intent(self, intent_name: str):
        """
        Removes intent from the database and then trains the model

        Args:
            intent_name: intent name

        """
        self.storage.remove_intent(intent_name)
        self.train()

    def add_phrase(self, intent_name: str, phrase: str):
        """
        Adds `phrase` to intent dictionary and then trains the model

        Args:
            intent_name: intent name
            phrase: phrase

        """
        self.storage.add_phrase(intent_name, phrase)
        self.train()

    def remove_phrase(self, intent_name: str, phrase: str):
        """
        Removes `phrase` from intent dictionary and then trains the model

        Args:
            intent_name: intent name
            phrase: phrase

        """
        self.storage.remove_phrase(intent_name, phrase)
        self.train()

    def change_answer(self, intent_name: str, answer: str):
        """
        Changes answer in intent dictionary and then trains the model

        Args:
            intent_name: intent name
            answer: answer

        """
        self.storage.change_answer(intent_name, answer)
        self.train()

    def check(self, context: UserContext) -> bool:
        """
        Checks:
         - if the handler function should be triggered based on the given context via context condition.
         - if at least one of the FAQ intents is matched to the `context.message`.

        Args:
            context: user context

        Returns:
            True, if handler should be activated, False otherwise
        """
        is_previous_matches = super().check(context)
        if not is_previous_matches:
            return False

        message = ' '.join(context.message)
        results = self.faq_model([message])
        results = list(zip(*results))
        handler_intents = self.storage.get_handler_intents()
        for option, score in results:
            (intent_name, intent_body) = list(json.loads(option).items())[0]
            if score > self.score_threshold and intent_name in handler_intents:
                return True
        return False

    def expand_context(self, context: UserContext) -> UserContext:
        context.handler_payload = {'faq_options': []}

        message = ' '.join(context.message)
        results = self.faq_model([message])
        results = list(zip(*results))
        handler_intents = self.storage.get_handler_intents()
        for option, score in results:
            (intent_name, intent_body) = list(json.loads(option).items())[0]
            if score > self.score_threshold and intent_name in handler_intents:
                context.handler_payload['faq_options'].append((intent_name, intent_body, score))
        if context.handler_payload['faq_options']:
            context.handler_payload['faq_options'].sort(key=lambda x: x[2], reverse=True)
        return context
