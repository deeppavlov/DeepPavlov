from typing import List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('dialogs_persona_parser')
class DialogsPersonaParser(Component):
    def __init__(self, **kwargs):
        pass

    def __call__(self, dialogs: List[dict]) -> List[List[str]]:
        bot_personas = []

        for dialog in dialogs:
            bot_personas.append(dialog['bot']['persona'])
        return bot_personas
