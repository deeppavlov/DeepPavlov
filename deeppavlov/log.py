import pathlib
import json
import logging.config


LOG_CONFIG_FILENAME = 'log_config.json'
DEFAULT_LOGGER_NAME = '_default'


def get_logger(logger_name):
    config_dir = pathlib.PurePath(__file__).parent
    log_config_path = pathlib.Path(config_dir, LOG_CONFIG_FILENAME).resolve()

    with open(log_config_path) as log_config_json:
        log_config = json.load(log_config_json)

    logging.config.dictConfig(log_config)

    try:
        logger = logging.getLogger(logger_name)
    except Exception:
        logger = logging.getLogger('_default')
        logger.error(
            'Can not initialise {} logger, logging to the stderr with {}'.format(logger_name, DEFAULT_LOGGER_NAME))

    return logger
