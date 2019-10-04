import threading
from logging import getLogger
from pathlib import Path
from queue import Empty, Queue
from threading import Thread, Timer
from typing import Union

from deeppavlov.core.commands.infer import build_model
from deeppavlov.deprecated.agents.default_agent import DefaultAgent
from deeppavlov.deprecated.agents.processors import DefaultRichContentWrapper
from deeppavlov.deprecated.skills.default_skill import DefaultStatelessSkill

log = getLogger(__name__)


class BaseBot(Thread):
    _config: dict
    input_queue: Queue
    _run_flag: bool
    _agent: DefaultAgent

    def __init__(self, model_config: Union[str, Path, dict],
                 default_skill_wrap: bool,
                 config: dict,
                 input_queue: Queue) -> None:
        super(BaseBot, self).__init__()
        self._config = config
        self.input_queue = input_queue
        self._run_flag = True

        model = build_model(model_config)
        skill = DefaultStatelessSkill(model) if default_skill_wrap else model
        self._agent = DefaultAgent([skill], skills_processor=DefaultRichContentWrapper())
        log.info('New bot instance level agent initiated')

    def run(self) -> None:
        """Thread run method implementation."""
        while self._run_flag:
            try:
                request = self.input_queue.get(timeout=1)
            except Empty:
                pass
            else:
                response = self._handle_request(request)
                self._send_response(response)

    def join(self, timeout=None):
        """Thread join method implementation."""
        self._run_flag = False
        for timer in threading.enumerate():
            if isinstance(timer, Timer):
                timer.cancel()
        Thread.join(self, timeout)

    def _handle_request(self, request: dict) -> dict:
        raise NotImplementedError

    def _send_response(self, response: dict) -> None:
        raise NotImplementedError
