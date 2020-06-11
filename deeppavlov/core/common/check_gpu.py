# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from logging import getLogger

import tensorflow as tf
from tensorflow.python.client import device_lib

log = getLogger(__name__)

_gpu_available = None


def check_gpu_existence():
    r"""Return True if at least one GPU is available"""
    global _gpu_available
    if _gpu_available is None:
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        try:
            with tf.Session(config=sess_config):
                device_list = device_lib.list_local_devices()
                _gpu_available = any(device.device_type == 'GPU' for device in device_list)
        except AttributeError as e:
            log.warning(f'Got an AttributeError `{e}`, assuming documentation building')
            _gpu_available = False
    return _gpu_available
