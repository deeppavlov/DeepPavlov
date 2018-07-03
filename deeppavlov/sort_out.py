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

import argparse
import sys

from pathlib import Path
from deeppavlov.pipeline_manager.pipeline_manager import PipelineManager
from deeppavlov.core.common.log import get_logger
from deeppavlov.download import deep_download

p = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(p))

log = get_logger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("mode", help="select a mode, train or interact", type=str,
                    choices={'train', 'evaluate', 'interact', 'predict', 'interactbot', 'riseapi', 'download'})
parser.add_argument("config_path", help="path to a pipeline json config", type=str)
parser.add_argument("exp_name", help="name of experiment", type=str)

parser.add_argument("-r", "--root", dest="root", default='./experiments',
                    help="folder where you will save the results and control points", type=str)
parser.add_argument("-hp", "--hyper", dest="hyper_search", default='random',
                    help="type of hyper search 'grid' or 'random'", type=str)
parser.add_argument("-sn", "--sample-num", dest="sample_num", default=10,
                    help="Number of generated samples if you use random search", type=int)
parser.add_argument("-tm", "--target-metric", dest="target_metric", default=None,
                    help="If you use more than one metric then target metric will be used"
                         " for results sortings", type=str)

parser.add_argument("-f", "--input-file", dest="file_path", default=None, help="Path to the input file", type=str)
parser.add_argument("-d", "--download", action="store_true", help="download model components")


def find_config(pipeline_config_path: str):
    if not Path(pipeline_config_path).is_file():
        configs = [c for c in Path(__file__).parent.glob(f'configs/**/{pipeline_config_path}.json')
                   if str(c.with_suffix('')).endswith(pipeline_config_path)]  # a simple way to not allow * and ?
        if configs:
            log.info(f"Interpreting '{pipeline_config_path}' as '{configs[0]}'")
            pipeline_config_path = str(configs[0])
    return pipeline_config_path


def main():
    args = parser.parse_args()
    pipeline_config_path = find_config(args.config_path)
    if args.download or args.mode == 'download':
        deep_download(['-c', pipeline_config_path])

    manager = PipelineManager(config_path=pipeline_config_path, exp_name=args.exp_name, mode=args.mode, root=args.root,
                              hyper_search=args.hyper_search, sample_num=args.sample_num, add_watcher=False,
                              target_metric=args.target_metric)
    manager.run()


if __name__ == "__main__":
    main()
