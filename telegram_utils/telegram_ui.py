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


def init_bot_for_model(token, model):
    bot = telebot.TeleBot(token)

    model_name = type(model).__name__
    config_path = Path(__file__).parent / 'models_info.json'
    models_info = read_json(str(config_path))
    model_info = models_info[model_name] if model_name in models_info else models_info['@default']

    @bot.message_handler(commands=['start'])
    def send_start_message(message):
        chat_id = message.chat.id
        out_message = model_info['start_message']
        if hasattr(model, 'reset'):
            model.reset()
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

        pred = model([context])
        reply_message = str(pred[0])
        bot.send_message(chat_id, reply_message)

    bot.polling()


def interact_model_by_telegram(config_path, token):
    config = read_json(config_path)
    model = build_model_from_config(config)
    init_bot_for_model(token, model)
