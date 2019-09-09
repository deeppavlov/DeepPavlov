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
import logging
import sqlite3
from typing import Union, NamedTuple, Optional, Iterable

UserId = Union[str, int]


class SkillResponse(NamedTuple):
    response: str
    confidence: float


class DummyLogger(logging.Logger):
    """
    Logger that does nothing.
    """

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        return None


def execute_query(connection_string: str,
                  sql_query: str,
                  params: Optional[Iterable] = None,
                  logger: Optional[logging.Logger] = None):
    logger = logger or DummyLogger('dummy')
    if connection_string:
        connection = sqlite3.connect(connection_string)
        try:
            cursor = connection.cursor()
            if params:
                cursor.execute(sql_query, params)
            else:
                cursor.execute(sql_query)
            connection.commit()
            return cursor
        except sqlite3.OperationalError:
            connected = False
            retries = 0
            while not connected and retries <= 12:  # try to establish a connection for no more than 1 min
                try:
                    sqlite3.connect(connection_string, timeout=5)
                    connected = True
                except:
                    logger.error("Reconnecting...")
                    retries = retries + 1
            return execute_query(connection_string, sql_query, params, logger)
        except Exception as e:
            logger.error(e)

