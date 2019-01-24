# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Dict, List, Union

import matplotlib
import numpy as np

matplotlib.use('agg', warn=False, force=True)


def get_met_info(logs_: List[Dict]) -> Dict:
    """
    Retrieves the necessary information from the log to build a histogram of results.

    Args:
        logs_: experiment logs

    Returns:
        data: experiments logs grouped by model name
    """
    main = dict()

    if logs_[0]['results'].get('test'):
        metrics_ = list(logs_[0]['results']['test'].keys())
    else:
        metrics_ = list(logs_[0]['results']['valid'].keys())

    group_data = dict()
    for val in logs_:
        if val['model'] not in group_data:
            group_data[val['model']] = dict()
            group_data[val['model']][val['pipe_index']] = val
        else:
            group_data[val['model']][val['pipe_index']] = val

    for name in group_data.keys():
        main[name] = dict()
        for met in metrics_:
            met_max = -1
            for key, val in group_data[name].items():
                if val['results'].get('test') is not None:
                    if val['results']['test'][met] > met_max:
                        met_max = val['results']['test'][met]
                else:
                    if val['results'].get('valid'):
                        if val['results']['valid'][met] > met_max:
                            met_max = val['results']['valid'][met]
                    else:
                        raise ValueError("Pipe with number {0} not contain 'test' or 'valid' keys in results, "
                                         "and it will not participate in comparing the results to display "
                                         "the final plot.".format(key))
            main[name][met] = met_max
    return main


def plot_res(info: dict,
             name: str,
             savepath: Union[str, Path],
             save: bool = True,
             width: float = 0.2,
             fheight: int = 8,
             fwidth: int = 12,
             ext: str = 'png') -> None:
    """
    Creates a histogram with the results of various models based on the experiment log.

    Args:
        info: data for the plot
        name: name of the plot
        savepath: path to the folder where plot will be saved
        save: determine save plot or show it without saving
        width: width between columns of histogram
        fheight: height of the plot
        fwidth: width of the plot
        ext: extension of the plot file

    Returns:
        None
    """
    from matplotlib import pyplot as plt

    # prepeare data
    bar_list = []
    models = list(info.keys())
    metrics = list(info[models[0]].keys())
    n = len(metrics)

    for met in metrics:
        tmp = []
        for model in models:
            tmp.append(info[model][met])
        bar_list.append(tmp)

    x = np.arange(len(models))

    # ploting
    fig, ax = plt.subplots()
    fig.set_figheight(fheight)
    fig.set_figwidth(fwidth)

    colors = plt.get_cmap('Paired')
    colors = colors(np.linspace(0, 0.5, len(bar_list)))
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores').set_fontsize(20)
    ax.set_title('Scores by metric').set_fontsize(20)

    bars = []
    for i, y in enumerate(bar_list):
        if i == 0:
            bars.append(ax.bar(x, y, width, color=colors[i]))
        else:
            bars.append(ax.bar(x + i * width, y, width, color=colors[i]))

    # plot x sticks and labels
    ax.set_xticks(x - width / 2 + n * width / 2)
    ax.set_xticklabels(tuple(models), fontsize=15)

    yticks = ax.get_yticks()
    ax.set_yticklabels(['{0:.2}'.format(float(y)) for y in yticks], fontsize=15)

    ax.grid(True, linestyle='--', color='b', alpha=0.1)

    # plot legend
    # ax.legend(tuple([bar[0] for bar in bars]), tuple(metrics), loc='upper left', bbox_to_anchor=(1, 1))
    ax.legend(tuple([bar[0] for bar in bars]), tuple(metrics))

    # auto lables
    def autolabel(columns):
        for rects in columns:
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '{0:.2}'.format(float(height)),
                        ha='center', va='bottom', fontsize=12)

    autolabel(bars)
    plt.ylim(0, 1.1)

    # show the picture
    if not save:
        plt.show()
    else:
        if not savepath.is_dir():
            savepath.mkdir()
        adr = savepath / '{0}.{1}'.format(name, ext)
        fig.savefig(str(adr), dpi=100)
        plt.close(fig)
