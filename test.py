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

from pathlib import Path
import sys
import os

p = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(p))

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.pipeline_manager.pipeline_manager import PipelineManager
from deeppavlov.core.commands.infer import interact_model, predict_on_stream
from deeppavlov.core.common.log import get_logger
from deeppavlov.download import deep_download
from utils.telegram_utils.telegram_ui import interact_model_by_telegram
from utils.server_utils.server import start_model_server
from utils.pip_wrapper import install_from_config


log = get_logger(__name__)


def find_config(pipeline_config_path: str):
    if not Path(pipeline_config_path).is_file():
        configs = [c for c in Path(__file__).parent.glob(f'configs/**/{pipeline_config_path}.json')
                   if str(c.with_suffix('')).endswith(pipeline_config_path)]  # a simple way to not allow * and ?
        if configs:
            log.info(f"Interpreting '{pipeline_config_path}' as '{configs[0]}'")
            pipeline_config_path = str(configs[0])
    return pipeline_config_path


def main():
    args = {'config_path': './deeppavlov/configs/my_configs/intents/intents_snips.json',
            'mode': 'sort_out',
            'exp_name': 'test',
            'hyper_search': 'grid',
            'sample_num': 10,
            'target_metric': None,
            'root': './download/experiments',
            'download': '', 'token': '', 'batch_size': '', 'file_path': ''}
    pipeline_config_path = find_config(args['config_path'])
    if args['download'] or args['mode'] == 'download':
        deep_download(['-c', pipeline_config_path])
    token = args['token'] or os.getenv('TELEGRAM_TOKEN')

    if args['mode'] == 'train':
        train_evaluate_model_from_config(pipeline_config_path)
    elif args['mode'] == 'evaluate':
        train_evaluate_model_from_config(pipeline_config_path, to_train=False, to_validate=False)
    elif args['mode'] == 'interact':
        interact_model(pipeline_config_path)
    elif args['mode'] == 'interactbot':
        if not token:
            log.error('Token required: initiate -t param or TELEGRAM_BOT env var with Telegram bot token')
        else:
            interact_model_by_telegram(pipeline_config_path, token)
    elif args['mode'] == 'riseapi':
        start_model_server(pipeline_config_path)
    elif args['mode'] == 'predict':
        predict_on_stream(pipeline_config_path, args['batch_size'], args['file_path'])
    elif args['mode'] == 'sort_out':
        manager = PipelineManager(config_path=pipeline_config_path, exp_name=args['exp_name'], mode='train',
                                  root=args['root'], hyper_search=args['hyper_search'], sample_num=args['sample_num'],
                                  target_metric=args['target_metric'])
        manager.run()
    elif args['mode'] == 'install':
        install_from_config(pipeline_config_path)


if __name__ == "__main__":
    main()

# Thread 1 "python" received signal SIGSEGV, Segmentation fault.
# 0x00007fff8a3b4398 in tensorflow::TF_SessionReleaseCallable(TF_Session*, long, TF_Status*) () from /home/mks/projects/DeepPavlov/venv/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so

#0  0x00007fff8a3c7398 in tensorflow::TF_SessionReleaseCallable(TF_Session*, long, TF_Status*) () from /home/mks/projects/DeepPavlov/venv/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
#1  0x00007fff8a3765e2 in _wrap_TF_SessionReleaseCallable () from /home/mks/projects/DeepPavlov/venv/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
#2  0x00000000004c53cd in _PyCFunction_FastCallKeywords ()
#3  0x000000000054ffe4 in ?? ()
#4  0x00000000005546cf in _PyEval_EvalFrameDefault ()
#5  0x000000000054f0e8 in ?? ()
#6  0x0000000000558ef2 in _PyFunction_FastCallDict ()
#7  0x000000000045a461 in _PyObject_Call_Prepend ()
#8  0x000000000045a0e3 in _PyObject_FastCallDict ()
#9  0x00000000004e14d1 in ?? ()
#10 0x0000000000442e00 in ?? ()
#11 0x000000000044442a in _PyObject_GC_NewVar ()
#12 0x00000000004d6a1d in PyTuple_New ()
#13 0x00000000005550b2 in _PyEval_EvalFrameDefault ()
#14 0x000000000054f0e8 in ?? ()
#15 0x0000000000550116 in ?? ()
#16 0x00000000005546cf in _PyEval_EvalFrameDefault ()
#17 0x000000000054fbe1 in ?? ()
#18 0x0000000000558d76 in _PyFunction_FastCallDict ()
#19 0x000000000045a461 in _PyObject_Call_Prepend ()
#20 0x0000000000459eee in PyObject_Call ()
#21 0x00000000004e15bb in ?? ()
#22 0x00000000004db337 in ?? ()
#23 0x000000000045a0e3 in _PyObject_FastCallDict ()
#24 0x000000000045a79c in _PyObject_FastCallKeywords ()
#25 0x000000000054fd37 in ?? ()
#26 0x0000000000552b00 in _PyEval_EvalFrameDefault ()
#27 0x000000000054fbe1 in ?? ()
