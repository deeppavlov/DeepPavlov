from pathlib  import Path

from deeppavlov.common import paths
from deeppavlov.common.file import read_json
from deeppavlov.common.params import from_params
from deeppavlov.common.registry import _REGISTRY


def interact(model):
    model.reset()

    while True:
        # get input from user
        utt = input(':: ')

        # check if user wants to begin new session
        if utt == 'clear' or utt == 'reset' or utt == 'restart':
            model.reset()
            print('')

        # check for exit command
        elif utt == 'exit' or utt == 'stop' or utt == 'quit' or utt == 'q':
            break

        else:
            # ENTER press : silence
            if not utt:
                utt = '<SILENCE>'

            # forward
            pred = model.infer(utt)
            print('>>', model.action_tracker.action_templates[pred])


def infer(config_path, usr_dir_name='USR_DIR'):
    config = read_json(config_path)

    # make a serialization user dir
    root_ = Path(config_path).resolve().parent
    usr_dir_path = root_.joinpath(usr_dir_name)

    paths.USR_PATH = usr_dir_path

    vocab_path = Path(usr_dir_path).joinpath('vocab.txt')

    model_config = config['model']
    model_name = model_config['name']
    model = from_params(_REGISTRY[model_name], model_config, vocab_path=vocab_path)
    return interact(model)
