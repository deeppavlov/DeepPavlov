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

"""Request parameters for the DeepPavlov model launched as a skill for Yandex.Alice.

Request parameters from this module are used to declare additional information
and validation for request parameters to the DeepPavlov model launched as
a skill for Yandex.Alice.

See details at https://fastapi.tiangolo.com/tutorial/body-multiple-params/

"""

from fastapi import Body

_body_example = {
    'name': 'data',
    'in': 'body',
    'required': 'true',
    'example': {
        'meta': {
            'locale': 'ru-RU',
            'timezone': 'Europe/Moscow',
            "client_id": 'ru.yandex.searchplugin/5.80 (Samsung Galaxy; Android 4.4)'
        },
        'request': {
            'command': 'где ближайшее отделение',
            'original_utterance': 'Алиса спроси у Сбербанка где ближайшее отделение',
            'type': 'SimpleUtterance',
            'markup': {
                'dangerous_context': True
            },
            'payload': {}
        },
        'session': {
            'new': True,
            'message_id': 4,
            'session_id': '2eac4854-fce721f3-b845abba-20d60',
            'skill_id': '3ad36498-f5rd-4079-a14b-788652932056',
            'user_id': 'AC9WC3DF6FCE052E45A4566A48E6B7193774B84814CE49A922E163B8B29881DC'
        },
        'version': '1.0'
    }
}

data_body = Body(..., example=_body_example)
