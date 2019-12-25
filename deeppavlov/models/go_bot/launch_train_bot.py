import argparse
import json
import logging
from logging import getLogger
from pathlib import Path
from typing import Dict, Any

from deeppavlov import configs, evaluate_model, train_model
from deeppavlov.core.common.file import read_json
from deeppavlov.core.data.sqlite_database import Sqlite3Database
from deeppavlov.dataset_iterators.dialog_iterator import DialogDatasetIterator
from deeppavlov.dataset_readers.dstc2_reader import SimpleDSTC2DatasetReader
from deeppavlov.download import download_decompress

log = getLogger(__name__)
log.setLevel(logging.DEBUG)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(asctime)s] %(message)s",
    datefmt="%y-%m-%d %H:%M:%S"
)

formatter = logging.Formatter('[%(levelname)s] [%(asctime)s] %(message)s',"%y-%m-%d %H:%M:%S", "%")
fh = logging.FileHandler("log")
fh.setFormatter(formatter)
log.addHandler(fh)


def load_database(bot_data_dir: Path, data: Dict[str, Any]):
    database = Sqlite3Database(
        primary_keys=["name"],
        save_path=str(bot_data_dir / "db.sqlite")
    )

    iterator = DialogDatasetIterator(data)
    db_results = []

    for dialog in iterator.gen_batches(batch_size=1, data_type='all'):
        turns_x, turns_y = dialog
        db_results.extend(x['db_result'] for x in turns_x[0] if x.get('db_result'))

    print(f"Adding {len(db_results)} items.")
    if db_results:
        database.fit(db_results)


def load_bot_data(bot_data_dir: Path, max_train_data: int = None):
    data = SimpleDSTC2DatasetReader().read(str(bot_data_dir))

    if max_train_data is not None:
        src = bot_data_dir / 'simple-dstc2-trn.json'
        dst = bot_data_dir / 'simple-dstc2-trn_full.json'

        old_data = json.loads(src.read_text(encoding='utf-8'))
        dst.write_text(json.dumps(old_data, indent=2), encoding='utf-8')
        src.write_text(json.dumps(old_data[:max_train_data], indent=2), encoding='utf-8')

        log.info(f"Train set is reduced to {max_train_data} dialogues (out of {len(data)}).")

    # load slot values
    download_decompress(
        url='http://files.deeppavlov.ai/deeppavlov_data/dstc_slot_vals.tar.gz',
        download_path=bot_data_dir
    )

    load_database(bot_data_dir, data)


def create_bot_configs(bot_data_dir: Path, bot_model_dir: Path, load_data: bool = False, max_train_data: int = None):
    if not bot_data_dir.exists() or load_data:
        load_bot_data(bot_data_dir, max_train_data)

    slotfill_config = read_json(configs.ner.slotfill_simple_dstc2_raw)
    slotfill_config['metadata']['variables']['DATA_PATH'] = str(bot_data_dir)

    slot_vals_file = 'dstc_slot_vals.json'
    slotfill_config['metadata']['variables']['SLOT_VALS_PATH'] = str(bot_data_dir / slot_vals_file)

    bot_model_dir.mkdir(parents=True, exist_ok=True)

    path_to_slotfill_config = bot_model_dir / 'slotfill_config.json'
    path_to_slotfill_config.write_text(json.dumps(slotfill_config, indent=2), encoding='utf-8')
    log.info(f'slot_fill config is saved to {path_to_slotfill_config}')

    gobot_config = read_json(configs.go_bot.gobot_simple_dstc2)
    gobot_config['chainer']['pipe'][-1]['embedder'] = None

    db_path = 'db.sqlite'
    gobot_config['chainer']['pipe'][-1]['database'] = {
        'class_name': 'sqlite_database',
        'primary_keys': ['name'],
        'save_path': str(bot_data_dir / db_path)
    }

    gobot_config['chainer']['pipe'][-1]['slot_filler']['config_path'] = str(path_to_slotfill_config)
    gobot_config['chainer']['pipe'][-1]['tracker']['slot_names'] = ['pricerange', 'this', 'area', 'food']

    gobot_config['chainer']['pipe'][-1]['template_type'] = 'DefaultTemplate'
    gobot_config['chainer']['pipe'][-1]['template_path'] = str(bot_data_dir / 'simple-dstc2-templates.txt')

    gobot_config['metadata']['variables']['DATA_PATH'] = str(bot_data_dir)
    gobot_config['metadata']['variables']['MODEL_PATH'] = str(bot_model_dir)

    gobot_config['train']['epochs'] = 200
    gobot_config['train']['batch_size'] = 8
    gobot_config['train']['max_batches'] = 250
    gobot_config['train']['log_on_k_batches'] = 20
    gobot_config['train']['val_every_n_batches'] = 40
    gobot_config['train']['log_every_n_batches'] = 40

    path_to_gobot_config = bot_model_dir / 'gobot_config.json'
    path_to_gobot_config.write_text(json.dumps(gobot_config, indent=2), encoding='utf-8')
    log.info(f'gobot config is saved to {path_to_gobot_config}')

    return gobot_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bot_data_dir", default=None, type=Path, required=True,
                        help="Relative path from working directory to a directory where go-bot data will be saved")
    parser.add_argument("--bot_model_dir", default=None, type=Path, required=True,
                        help="Relative path from working directory to a directory where go-bot model will be saved")

    # Other parameters
    parser.add_argument("--working_directory", default="", type=Path, required=False,
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

    bot_data_dir = args.working_directory / args.bot_data_dir
    bot_model_dir = args.working_directory / args.bot_model_dir

    gobot_config = create_bot_configs(bot_data_dir, bot_model_dir, args.load_data, args.max_train_data)

    if args.do_train:
        train_model(gobot_config)

    if args.do_eval:
        evaluate_model(gobot_config)
