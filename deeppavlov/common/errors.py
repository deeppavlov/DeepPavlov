import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """
    Any configuration error.
    """

    def __init__(self, message):
        super(ConfigError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)
