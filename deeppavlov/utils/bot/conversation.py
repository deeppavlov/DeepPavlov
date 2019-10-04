from threading import Timer

from deeppavlov.deprecated.agents.default_agent import DefaultAgent


class BaseConversation:
    config: dict
    agent: DefaultAgent
    conversation_lifetime: float
    timer: Timer

    def __init__(self, config: dict, agent: DefaultAgent, conversation_key, self_destruct_callback: callable):
        self.config = config
        self.agent = agent
        self.key = conversation_key
        self._self_destruct_callback = self_destruct_callback
        self.conversation_lifetime = self.config['conversation_lifetime']

    def _start_timer(self) -> None:
        """Initiates self-destruct timer."""
        self.timer = Timer(self.config['conversation_lifetime'], self._self_destruct_callback)
        self.timer.start()

    def _rearm_self_destruct(self) -> None:
        """Rearms self-destruct timer."""
        self.timer.cancel()
        self._start_timer()
