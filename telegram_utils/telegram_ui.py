import telebot

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config


def init_bot_for_model(token, model):
    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        chat_id = message.chat.id
        bot.send_message(chat_id, 'Welcome to the DeepPavlov inference bot!')

    @bot.message_handler()
    def handle_inference(message):
        chat_id = message.chat.id
        context = message.text

        pred = model.infer(context)
        reply_message = 'model prediction: {}'.format(str(pred))
        bot.send_message(chat_id, reply_message)

    bot.polling()


def interact_model_by_telegram(config_path, token):
    config = read_json(config_path)
    model = build_model_from_config(config)
    init_bot_for_model(token, model)
