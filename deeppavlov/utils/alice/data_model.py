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
