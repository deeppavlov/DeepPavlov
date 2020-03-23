from deeppavlov.dataset_readers.dstc2_reader import SimpleDSTC2DatasetReader
import os
from deeppavlov.dataset_iterators.dialog_iterator import DialogDatasetIterator
from pprint import pprint
import shutil
import json
from deeppavlov.core.data.sqlite_database import Sqlite3Database
from deeppavlov import configs
from deeppavlov.core.common.file import read_json
from deeppavlov import evaluate_model
from deeppavlov import build_model
from deeppavlov import train_model
from deeppavlov.download import download_decompress

base_working_dir = "gobot_runner_dir"
try:
    os.mkdir(base_working_dir)
except FileExistsError:
    pass

os.chdir(base_working_dir)

data = SimpleDSTC2DatasetReader.read("my_data")

iterator = DialogDatasetIterator(data)

for dialog in iterator.gen_batches(batch_size=1, data_type='train'):
    turns_x, turns_y = dialog

    print("User utterances:\n----------------\n")
    pprint(turns_x[0], indent=4)
    print("\nSystem responses:\n-----------------\n")
    pprint(turns_y[0], indent=4)

    break

shutil.copyfile("my_data/simple-dstc2-trn.json", "my_data/simple-dstc2-trn.full.json")

NUM_TRAIN = 50

with open('my_data/simple-dstc2-trn.full.json', 'rt') as fin:
    data = json.load(fin)
with open('my_data/simple-dstc2-trn.json', 'wt') as fout:
    json.dump(data[:NUM_TRAIN], fout, indent=2)
print(f"Train set is reduced to {NUM_TRAIN} dialogues (out of {len(data)}).")

database = Sqlite3Database(primary_keys=["name"],
                           save_path="my_bot/db.sqlite")

db_results = []

for dialog in iterator.gen_batches(batch_size=1, data_type='all'):
    turns_x, turns_y = dialog
    db_results.extend(x['db_result'] for x in turns_x[0] if x.get('db_result'))

print(f"Adding {len(db_results)} items.")
if db_results:
    database.fit(db_results)

print(database([{'pricerange': 'cheap', 'area': 'south'}]))

download_decompress(url='http://files.deeppavlov.ai/deeppavlov_data/dstc_slot_vals.tar.gz',
                    download_path='my_bot/slotfill')

slotfill_config = read_json(configs.ner.slotfill_simple_dstc2_raw)

slotfill_config['metadata']['variables']['DATA_PATH'] = 'my_data'
slotfill_config['metadata']['variables']['SLOT_VALS_PATH'] = 'my_bot/slotfill/dstc_slot_vals.json'

slotfill = evaluate_model(slotfill_config);

slotfill = build_model(slotfill_config)

slotfill(['i want cheap chinee food'])

import json

json.dump(slotfill_config, open('my_bot/slotfill_config.json', 'wt'))

gobot_config = read_json(configs.go_bot.gobot_simple_dstc2)

gobot_config['chainer']['pipe'][-1]['embedder'] = None

gobot_config['chainer']['pipe'][-1]['database'] = {
    'class_name': 'sqlite_database',
    'primary_keys': ["name"],
    'save_path': 'my_bot/db.sqlite'
}

gobot_config['chainer']['pipe'][-1]['slot_filler']['config_path'] = 'my_bot/slotfill_config.json'

gobot_config['chainer']['pipe'][-1]['tracker']['slot_names'] = ['pricerange', 'this', 'area', 'food']

gobot_config['chainer']['pipe'][-1]['template_type'] = 'DefaultTemplate'
gobot_config['chainer']['pipe'][-1]['template_path'] = 'my_data/simple-dstc2-templates.txt'

gobot_config['metadata']['variables']['DATA_PATH'] = 'my_data'
gobot_config['metadata']['variables']['MODEL_PATH'] = 'my_bot'

gobot_config['train']['batch_size'] = 8  # batch size
gobot_config['train']['max_batches'] = 250  # maximum number of training batches
gobot_config['train']['log_on_k_batches'] = 20
gobot_config['train']['val_every_n_batches'] = 40  # evaluate on full 'valid' split each n batches
gobot_config['train']['log_every_n_batches'] = 40  # evaluate on 20 batches of 'train' split every n batches

json.dump(gobot_config, open('my_bot/gobot_config.json', 'wt'))


train_model(gobot_config);

evaluate_model(gobot_config);

bot = build_model(gobot_config)

bot(['hi, i want to eat, can you suggest a place to go?'])
