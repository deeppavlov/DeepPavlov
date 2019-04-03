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

from deeppavlov.core.common.paths import get_settings_path, populate_settings_dir

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--default", action="store_true", help="return to defaults")


def main():
    """DeepPavlov console configuration utility."""
    args = parser.parse_args()
    path = get_settings_path()

    if args.default:
        if populate_settings_dir(force=True):
            print(f'Populated {path} with default settings files')
        else:
            print(f'{path} is already a default settings directory')
    else:
        print(f'Current DeepPavlov settings path: {path}')


if __name__ == "__main__":
    main()
