import pathlib
import json
import logging.config
import sys


LOG_CONFIG_FILENAME = 'log_config.json'


def get_logger(logger_name):
    try:
        config_dir = pathlib.PurePath(__file__).parent
        log_config_path = pathlib.Path(config_dir, '..', '..', LOG_CONFIG_FILENAME).resolve()

        with open(log_config_path) as log_config_json:
            log_config = json.load(log_config_json)

        logging.config.dictConfig(log_config)
        logger = logging.getLogger(logger_name)

    except Exception:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)

        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)d %(levelname)s in \'%(module)s\' at line %(lineno)d: %(message)s',
            '%Y-%m-%d %H:%M:%S')

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        handler.setLevel(logging.WARNING)

        logger.addHandler(handler)

        logger.error(
            'LOGGER ERROR: Can not initialise {} logger, '
            'logging to the stderr. Error traceback:\n'.format(logger_name), exc_info=1)

    return logger
