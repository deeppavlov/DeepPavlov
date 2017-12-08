from deeppavlov.models.inferable import Inferable
from deeppavlov.common.registry import register_model

@register_model("dummy")
class DummySkill(Inferable):
    def __init__(self, vocab_path=None):
        pass

    def infer(self, instance):
        return 'Hello, my name is Kaspar Hauser'

    def reset(self):
        pass
