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

"""Data models for the DeepPavlov model launched as a skill for Yandex.Alice.

Data models from this module are used to declare the body of the request
to the DeepPavlov model launched as a skill for Yandex.Alice. Data models
are declared as a classes that inherits from Pydantic Base Model. FastAPI uses
this models to:

    * read the body of the request as JSON
    * convert the corresponding types (if needed)
    * validate the data
    * generate JSON schemas used by automatic documentation UIs

See details at https://fastapi.tiangolo.com/tutorial/body/,
               https://fastapi.tiangolo.com/tutorial/body-nested-models/

"""
from typing import Dict

from fastapi import Body
from pydantic import BaseModel


class Meta(BaseModel):
    locale: str = Body(..., example='ru-RU')
    timezone: str = Body(..., example='Europe/Moscow')
    client_id: str = Body(..., example='ru.yandex.searchplugin/5.80 (Samsung Galaxy; Android 4.4)')


class Markup(BaseModel):
    dangerous_context: bool = Body(..., example=True)


class Request(BaseModel):
    command: str = Body(..., example='где ближайшее отделение')
    original_utterance: str = Body(..., example='Алиса спроси у Сбербанка где ближайшее отделение')
    type: str = Body(..., example='SimpleUtterance')
    markup: Markup
    payload: Dict = Body(..., example={})


class Session(BaseModel):
    new: bool = Body(..., example=True)
    message_id: int = Body(..., example=4)
    session_id: str = Body(..., example='2eac4854-fce721f3-b845abba-20d60')
    skill_id: str = Body(..., example='3ad36498-f5rd-4079-a14b-788652932056')
    user_id: str = Body(..., example='AC9WC3DF6FCE052E45A4566A48E6B7193774B84814CE49A922E163B8B29881DC')


class Data(BaseModel):
    meta: Meta
    request: Request
    session: Session
    version: str = Body(..., example='1.0')
