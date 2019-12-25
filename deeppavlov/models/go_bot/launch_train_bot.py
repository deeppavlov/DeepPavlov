import argparse
import json
import logging
import os
from logging import getLogger

from deeppavlov import configs, evaluate_model, train_model
from deeppavlov.core.common.file import read_json
from deeppavlov.core.data.sqlite_database import Sqlite3Database
from deeppavlov.dataset_iterators.dialog_iterator import DialogDatasetIterator
from deeppavlov.dataset_readers.dstc2_reader import SimpleDSTC2DatasetReader
from deeppavlov.download import download_decompress

log = getLogger(__name__)
log.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(levelname)s] [%(asctime)s] %(message)s',"%y-%m-%d %H:%M:%S", "%")
fh = logging.FileHandler("log")
fh.setFormatter(formatter)
log.addHandler(fh)


def load_database(bot_data_dir, data):
    database = Sqlite3Database(
        primary_keys=["name"],
        save_path=os.path.join(bot_data_dir, "db.sqlite")
    )

    iterator = DialogDatasetIterator(data)
    db_results = []

    for dialog in iterator.gen_batches(batch_size=1, data_type='all'):
        turns_x, turns_y = dialog
        db_results.extend(x['db_result'] for x in turns_x[0] if x.get('db_result'))

    print(f"Adding {len(db_results)} items.")
    if db_results:
        database.fit(db_results)


def load_bot_data(bot_data_dir, max_train_data):
    data = SimpleDSTC2DatasetReader().read(bot_data_dir)

    if max_train_data is not None:
        src = os.path.join(bot_data_dir, 'simple-dstc2-trn.json')
        dst = os.path.join(bot_data_dir, 'simple-dstc2-trn_full.json')

        with open(src, 'rt') as fin:
            old_data = json.load(fin)

        with open(dst, 'wt') as fout:
            json.dump(old_data, fout, indent=2)

        with open(src, 'wt') as fout:
            json.dump(old_data[:max_train_data], fout, indent=2)

        log.info(f"Train set is reduced to {max_train_data} dialogues (out of {len(data)}).")

    # load slot values
    download_decompress(
        url='http://files.deeppavlov.ai/deeppavlov_data/dstc_slot_vals.tar.gz',
        download_path=bot_data_dir
    )

    load_database(bot_data_dir, data)


def create_bot_configs(bot_data_dir, bot_model_dir, load_data=False, max_train_data=None):
    if not os.path.exists(bot_data_dir) or load_data:
        load_bot_data(bot_data_dir, max_train_data)

    slotfill_config = read_json(configs.ner.slotfill_simple_dstc2_raw)
    slotfill_config['metadata']['variables']['DATA_PATH'] = bot_data_dir

    slot_vals_file = 'dstc_slot_vals.json'
    slotfill_config['metadata']['variables']['SLOT_VALS_PATH'] = os.path.join(
        bot_data_dir, slot_vals_file
    )

    if not os.path.exists(bot_model_dir):
        os.makedirs(bot_model_dir)

    path_to_slotfill_config = os.path.join(bot_model_dir, 'slotfill_config.json')
    with open(path_to_slotfill_config, 'w') as f:
        json.dump(slotfill_config, f, indent=2)
        log.info(f'slot_fill config is saved to {path_to_slotfill_config}')

    gobot_config = read_json(configs.go_bot.gobot_simple_dstc2)
    gobot_config['chainer']['pipe'][-1]['embedder'] = None

    db_path = 'db.sqlite'
    gobot_config['chainer']['pipe'][-1]['database'] = {
        'class_name': 'sqlite_database',
        'primary_keys': ['name'],
        'save_path': os.path.join(bot_data_dir, db_path)
    }

    gobot_config['chainer']['pipe'][-1]['slot_filler']['config_path'] = path_to_slotfill_config

    slot_names_dstc2 = ['pricerange', 'this', 'area', 'food']

    gobot_config['chainer']['pipe'][-1]['tracker']['slot_names'] = slot_names_dstc2

    gobot_config['chainer']['pipe'][-1]['template_type'] = 'DefaultTemplate'
    gobot_config['chainer']['pipe'][-1]['template_path'] = os.path.join(bot_data_dir, 'simple-dstc2-templates.txt')

    gobot_config['metadata']['variables']['DATA_PATH'] = bot_data_dir
    gobot_config['metadata']['variables']['MODEL_PATH'] = bot_model_dir

    gobot_config['train']['epochs'] = 200
    gobot_config['train']['batch_size'] = 8
    gobot_config['train']['max_batches'] = 250
    gobot_config['train']['log_on_k_batches'] = 20
    gobot_config['train']['val_every_n_batches'] = 40
    gobot_config['train']['log_every_n_batches'] = 40

    path_to_gobot_config = os.path.join(bot_model_dir, 'gobot_config.json')
    with open(path_to_gobot_config, 'w') as f:
        json.dump(gobot_config, f, indent=2)
        log.info(f'gobot config is saved to {path_to_gobot_config}')

    return gobot_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bot_data_dir", default=None, type=str, required=True,
                        help="Relative path from working directory to a directory where go-bot data will be saved")
    parser.add_argument("--bot_model_dir", default=None, type=str, required=True,
                        help="Relative path from working directory to a directory where go-bot model will be saved")

    # Other parameters
    parser.add_argument("--working_directory", default="", type=str, required=False,
                        help="Directory from which relative paths are built")
    parser.add_argument("--load_data", action='store_true',
                        help="Whether to load data")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to train the bot")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to evaluate the bot performance.")
    parser.add_argument("--max_train_data", default=None, type=int, required=False,
                        help="Maximum number of dialogues to take from training data")

    args = parser.parse_args()

    bot_data_dir = os.path.join(args.working_directory, args.bot_data_dir)
    bot_model_dir = os.path.join(args.working_directory, args.bot_model_dir)

    gobot_config = create_bot_configs(bot_data_dir, bot_model_dir, args.load_data, args.max_train_data)

    if args.do_train:
        train_model(gobot_config)

    if args.do_eval:
        evaluate_model(gobot_config)
