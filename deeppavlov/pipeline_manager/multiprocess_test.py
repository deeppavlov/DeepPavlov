import argparse

from deeppavlov.pipeline_manager.pipeline_manager_parallel import PipelineManager
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("-tm", "--target-metric", dest="target_metric", default=None,
                    help="If you use more than one metric then target metric will be used"
                         " for results sortings", type=str)
parser.add_argument("config_path", help="path to a pipeline json config", type=str)


def find_config(pipeline_config_path: str):
    if not Path(pipeline_config_path).is_file():
        configs = [c for c in Path(__file__).parent.glob(f'configs/**/{pipeline_config_path}.json')
                   if str(c.with_suffix('')).endswith(pipeline_config_path)]  # a simple way to not allow * and ?
        if configs:
            pipeline_config_path = str(configs[0])
    return pipeline_config_path


if __name__ == '__main__':
    args = parser.parse_args()
    pipeline_config_path = find_config(args.config_path)
    target_metric = args.target_metric

    exp_name = 'multiprocess_test'
    root = './download/experiments'
    sample_num = 10

    manager = PipelineManager(config_path=pipeline_config_path, exp_name=exp_name, root=root, cross_val=False,
                              sample_num=sample_num, target_metric=target_metric, plot=False, save_best=True,
                              multiprocessing=True, use_all_gpus=True, use_multi_gpus=None, max_num_workers=None)
    manager.run()
