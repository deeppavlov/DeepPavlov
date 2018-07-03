from deeppavlov.pipeline_manager.pipegen import PipeGen


path = '/home/mks/projects/DeepPavlov/deeppavlov/configs/my_configs/intents/intents_snips.json'
exp_name = 'test'
mode = 'train'
root = '/home/mks/projects/DeepPavlov/deeppavlov/experiments/'
hyper_search = 'random'
sample_num = 10
target_metric = None

gen = PipeGen(config_path=path)

for i, pipe in enumerate(gen()):
    print(i)
    print(pipe)
