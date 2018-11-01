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

import argparse
import os
from pathlib import Path

from deeppavlov.core.common.paths import get_configs_path, set_configs_path, set_configs_default


parser = argparse.ArgumentParser()

parser.add_argument("mode", help="select DeepPavlov configuration option", type=str, choices={'settings'})

parser.add_argument("-p", "--path", default=None, help="set path", type=str)
parser.add_argument("-d", "--default", action="store_true", help="return to defaults")


def main():
    """DeepPavlov console configuration utility."""
    args = parser.parse_args()
    path = args.path

    if args.mode == 'settings':
        if args.default:
            set_configs_default()
        else:
            if not path:
                print(f'Current DeepPavlov settings path: {get_configs_path()}')
            else:
                path = Path(os.getcwd(), path).resolve()
                set_configs_path(path)


if __name__ == "__main__":
    main()
