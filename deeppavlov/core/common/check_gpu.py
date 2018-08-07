"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import tensorflow as tf
from tensorflow.python.client import device_lib

from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)


def _check_gpu_existence():
    r"""Return True if at least one GPU available"""
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    try:
        with tf.Session(config=sess_config):
            device_list = device_lib.list_local_devices()
            return any(device.device_type == 'GPU' for device in device_list)
    except AttributeError as e:
        log.warning(f'Got an AttributeError `{e}`, assuming documentation building')
        return False


GPU_AVAILABLE = _check_gpu_existence()
