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

from functools import wraps

from typing import Type, Callable

from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


class abstract_attribute(object):
    def __get__(self, obj, t: Type):
        for cls in type.__mro__:
            for name, value in cls.__dict__.items():
                if value is self:
                    this_obj = obj if obj else t
                    raise NotImplementedError(
                        '{} does not have the attribute {} '
                        '(abstract from class {}'.format(this_obj,
                                                         name,
                                                         cls.__name__))
            raise NotImplementedError('{} does not set the abstract attribute <unknown>'
                                      .format(t.__name__))


def check_attr_true(attr: str):
    def _check_attr_true(f: Callable):
        @wraps(f)
        def wrapped(self, *args, **kwargs):
            if getattr(self, attr):
                return f(self, *args, **kwargs)
            else:
                log.warning("'{0}' is False, doing nothing."
                            " Set '{0}' to True in json config "
                            "if you'd like the {1} to proceed.".format(attr, str(f).split()[1]))

        return wrapped

    return _check_attr_true

# def run_alt_meth_if_no_path(alt_f: Callable, attr: str):
#     def _run_alt_meth(f):
#         @wraps(f)
#         def wrapped(self, *args, **kwargs):
#             if self.ser_path.exists():
#                 if self.ser_path.is_file() or (
#                             self.ser_path.is_dir() and os.listdir(str(self.ser_path))):
#                     try:
#                         return f(self, *args, **kwargs)
#                     except ConfigError:
#                         print('There are no needed model files')
#             setattr(self, attr, True)
#             print(
#                 "Attribute '{0}' is set to False, though the path doesn't exist or there"
#                 " is no ser data at the given path.\nCan't do {1}()."
#                 " Instead will do {2}()".format(attr, str(f).split()[1], str(alt_f).split()[1]))
#             return alt_f(self, *args, **kwargs)
#
#         return wrapped
#
#     return _run_alt_meth
#
#
# def check_path_exists():
#     def _check_path_exists(f: Callable):
#         @wraps(f)
#         def wrapped(self, *args, **kwargs):
#             if self.ser_path.is_dir():
#                 return f(self, *args, **kwargs)
#             elif self.ser_path.parent.exists():
#                 return f(self, *args, **kwargs)
#             raise FileNotFoundError(
#                 "{}.ser_path doesn't exist. Check if there is a pretrained model."
#                 "If there is no a pretrained model, you might want to set 'train_now' to true "
#                 "in the model json config and run training first.".format(self.__class__.__name__))
#
#         return wrapped
#
#     return _check_path_exists
