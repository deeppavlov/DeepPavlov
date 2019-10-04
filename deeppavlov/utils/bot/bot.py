import threading
from logging import getLogger
from pathlib import Path
from queue import Queue
from threading import Thread, Timer
from typing import Union, Optional

from deeppavlov.core.commands.infer import build_model
from deeppavlov.deprecated.agents.default_agent import DefaultAgent
from deeppavlov.deprecated.agents.processors import DefaultRichContentWrapper
from deeppavlov.deprecated.skills.default_skill import DefaultStatelessSkill

log = getLogger(__name__)


class BaseBot(Thread):
    model_config: Union[str, Path, dict]
    default_skill_wrap: bool
    config: dict
    input_queue: Queue
    _run_flag: bool
    agent: Optional[DefaultAgent]
    timer: Timer

    def __init__(self, model_config: Union[str, Path, dict],
                 default_skill_wrap: bool,
                 config: dict,
                 input_queue: Queue) -> None:
        super(BaseBot, self).__init__()
        self.model_config = model_config
        self.default_skill_wrap = default_skill_wrap
        self.config = config
        self.input_queue = input_queue
        self._run_flag = True
        self.agent = None

        if not self.config['multi_instance']:
            self.agent = self._get_default_agent()
            log.info('New bot instance level agent initiated')

    def _get_default_agent(self) -> DefaultAgent:
        """Creates skill agent."""
        # TODO: Decide about multi-instance mode necessity.
        # If model multi-instancing is still necessary - refactor and remove
        model = build_model(self.model_config)
        skill = DefaultStatelessSkill(model) if self.default_skill_wrap else model
        agent = DefaultAgent([skill], skills_processor=DefaultRichContentWrapper())
        return agent

    def join(self, timeout=None):
        """Thread join method implementation."""
        self._run_flag = False
        for timer in threading.enumerate():
            if isinstance(timer, Timer):
                timer.cancel()
        Thread.join(self, timeout)
