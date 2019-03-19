"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import time
from logging import getLogger
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from pathlib import Path
from threading import Thread
from typing import Callable, Optional, Collection, Hashable, List, Tuple

import telebot
from telebot.types import Message

from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.default_rich_content_processor import DefaultRichContentWrapper
from deeppavlov.core.agent import Agent
from deeppavlov.core.agent.rich_content import RichMessage
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.skills.default_skill.default_skill import DefaultStatelessSkill

log = getLogger(__name__)

SERVER_CONFIG_FILENAME = 'server_config.json'
TELEGRAM_MODELS_INFO_FILENAME = 'models_info.json'


def _model_process(model_function: Callable, conn: Connection, batch_size: int = -1, *,
                   poll_period: float = 0.5):
    model: Callable[[Collection[str], Collection[Hashable]], Collection[str]] = model_function()
    if batch_size <= 0:
        batch_size = float('inf')

    while True:
        batch: List[Tuple[str, Hashable]] = []
        while conn.poll() and len(batch) < batch_size:
            batch.append(conn.recv())

        if not batch:
            time.sleep(poll_period)
            continue

        messages, dialog_ids = zip(*batch)
        responses = model(messages, dialog_ids)
        for response, dialog_id in zip(responses, dialog_ids):
            conn.send((response, dialog_id))


def experimental_bot(model_function: Callable[..., Callable[[Collection[Message], Collection[Hashable]], Collection[str]]],
                     token: str, proxy: Optional[str] = None, *, batch_size: int = -1, poll_period: float = 0.5):
    """

    Args:
        model_function: a function that produces an agent
        token: telegram token string
        proxy: https or socks5 proxy string for telebot
        batch_size: maximum batch size for the model
        poll_period: how long to wait every time no input was done for the model

    Returns: None

    """
    if proxy is not None:
        telebot.apihelper.proxy = {'https': proxy}

    bot = telebot.TeleBot(token)

    parent_conn, child_conn = Pipe()
    p = Process(target=_model_process, args=(model_function, child_conn),
                kwargs={'batch_size': batch_size, 'poll_period': poll_period})
    p.start()

    def responder():
        while True:
            text, chat_id = parent_conn.recv()
            bot.send_message(chat_id, text)

    t = Thread(target=responder)
    t.start()

    @bot.message_handler()
    def handle_message(message: Message):
        parent_conn.send((message, message.chat.id))

    bot.polling(none_stop=True)


def init_bot_for_model(agent: Agent, token: str, model_name: str):
    bot = telebot.TeleBot(token)

    models_info_path = Path(get_settings_path(), TELEGRAM_MODELS_INFO_FILENAME).resolve()
    models_info = read_json(str(models_info_path))
    model_info = models_info[model_name] if model_name in models_info else models_info['@default']

    @bot.message_handler(commands=['start'])
    def send_start_message(message):
        chat_id = message.chat.id
        out_message = model_info['start_message']
        bot.send_message(chat_id, out_message)

    @bot.message_handler(commands=['help'])
    def send_help_message(message):
        chat_id = message.chat.id
        out_message = model_info['help_message']
        bot.send_message(chat_id, out_message)

    @bot.message_handler()
    def handle_inference(message):
        chat_id = message.chat.id
        context = message.text

        response: RichMessage = agent([context], [chat_id])[0]
        for message in response.json():
            message_text = message['content']
            bot.send_message(chat_id, message_text)

    bot.polling()


def interact_model_by_telegram(config, token=None):
    server_config_path = Path(get_settings_path(), SERVER_CONFIG_FILENAME)
    server_config = read_json(server_config_path)
    token = token if token else server_config['telegram_defaults']['token']
    if not token:
        e = ValueError('Telegram token required: initiate -t param or telegram_defaults/token '
                       'in server configuration file')
        log.error(e)
        raise e

    model = build_model(config)
    model_name = type(model.get_main_component()).__name__
    skill = DefaultStatelessSkill(model)
    agent = DefaultAgent([skill], skills_processor=DefaultRichContentWrapper())
    init_bot_for_model(agent, token, model_name)
