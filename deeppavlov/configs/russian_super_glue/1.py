import json
path = "/cephfs/home/konovalov/DP6/DeepPavlov/deeppavlov/configs/russian_super_glue/russian_superglue_rcb_rubert.json"
with open(path, 'r') as f:
    data = json.load(f)
    with open('/tmp/config.json', 'w') as f:
        json.dump(data, f, sort_keys=False, indent=4, ensure_ascii=False)
