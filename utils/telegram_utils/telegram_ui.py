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
from pathlib import Path

import telebot

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.common.log import get_logger


TELEGRAM_UI_CONFIG_FILENAME = 'models_info.json'

log = get_logger(__name__)


def init_bot_for_model(token, model, model_config_key):
    bot = telebot.TeleBot(token)

    config_dir = Path(__file__).resolve().parent
    tg_config_path = Path(config_dir, TELEGRAM_UI_CONFIG_FILENAME).resolve()
    models_info = read_json(str(tg_config_path))

    if model_config_key in models_info:
        model_info = models_info[model_config_key]
    else:
        log.warn(f'Model config key "{model_config_key}" was not found in telegram_utils/models_info.json, '
                 'using default Telegram utils params')
        model_info = models_info['@default']

    buffer = {}
    expect = []

    def init_multiarg_infer(chat_id):
        buffer[chat_id] = []
        expect[:] = list(model.in_x)

    @bot.message_handler(commands=['start'])
    def send_start_message(message):
        chat_id = message.chat.id
        out_message = model_info['start_message']
        if hasattr(model, 'reset'):
            model.reset()
        bot.send_message(chat_id, out_message)
        if len(model.in_x) > 1:
            init_multiarg_infer(chat_id)
            bot.send_message(chat_id, f'Please, send {expect.pop(0)}')

    @bot.message_handler(commands=['help'])
    def send_help_message(message):
        chat_id = message.chat.id
        out_message = model_info['help_message']
        bot.send_message(chat_id, out_message)

    @bot.message_handler()
    def handle_inference(message):
        chat_id = message.chat.id
        context = message.text

        if len(model.in_x) > 1:
            if chat_id not in buffer:
                init_multiarg_infer(chat_id)

            buffer[chat_id].append(context)

            if expect:
                bot.send_message(chat_id, f'Please, send {expect.pop(0)}')
            else:
                pred = model([tuple(buffer[chat_id])])
                reply_message = str(pred[0])
                bot.send_message(chat_id, reply_message)

                buffer[chat_id] = []
                expect[:] = list(model.in_x)
                bot.send_message(chat_id, f'Please, send {expect.pop(0)}')
        else:
            pred = model([context])
            reply_message = str(pred[0])
            bot.send_message(chat_id, reply_message)

    bot.polling()


def interact_model_by_telegram(model_config_path, token):
    model_config = read_json(model_config_path)
    model_config_key = model_config['metadata']['labels']['telegram_utils']
    model = build_model_from_config(model_config)
    init_bot_for_model(token, model, model_config_key)
