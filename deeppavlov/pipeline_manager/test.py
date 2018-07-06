from deeppavlov.pipeline_manager.pipeline_manager import PipelineManager
from deeppavlov.core.commands.utils import set_deeppavlov_root, expand_path
from deeppavlov.download import deep_download

set_deeppavlov_root({})
data_path = expand_path('snips')

path = '/home/mks/projects/DeepPavlov/deeppavlov/configs/my_configs/intents/intents_snips.json'
exp_name = 'plot_test'
mode = 'train'
root = '/home/mks/projects/DeepPavlov/experiments/'
hyper_search = 'random'
sample_num = 10
target_metric = None


def main():

    deep_download(['-c', path])

    manager = PipelineManager(config_path=path, exp_name=exp_name, mode=mode, root=root,
                              hyper_search=hyper_search, sample_num=sample_num, add_watcher=False,
                              target_metric=target_metric)
    manager.run()


if __name__ == "__main__":
    main()
