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

from logging import getLogger
from typing import Iterable, List, Optional, Union

from py3nvml import py3nvml

from deeppavlov.core.common.errors import GpuError

logger = getLogger(__name__)


def get_available_gpus(num_gpus: Optional[int] = None,
                       gpu_select: Union[int, Iterable[int], None] = None,
                       gpu_fraction: float = 1.0) -> List[int]:
    """
    Considers all available to the user graphics cards. And selects as available only those that meet the memory
    criterion. If the memory of a video card is occupied by more than "X" percent, then the video card is considered
    inaccessible. For the value of the parameter "X" is responsible "gpu_fraction" argument.

    Args:
        num_gpus : int; How many gpus you need (optional)
        gpu_select : iterable; A single int or an iterable of ints indicating gpu numbers to search through.
         If left blank, will search through all gpus.
        gpu_fraction : float; The fractional of a gpu memory that must be free for the script to see the gpu as free.
         Defaults to 1. Useful if someone has grabbed a tiny amount of memory on a gpu but isn't using it.

    Returns:
        available_gpu: list of ints; List with available gpu numbers
    """
    # Try connect with NVIDIA drivers
    try:
        py3nvml.nvmlInit()
    except Exception:
        raise GpuError("Couldn't connect to nvidia drivers. Check they are installed correctly.")

    numdevices = py3nvml.nvmlDeviceGetCount()
    gpu_free = [False] * numdevices

    if num_gpus is None:
        num_gpus = numdevices
    elif num_gpus > numdevices:
        print("GpuWarning: Device have only {0} gpu cards, "
              "but parameter 'max_num_workers' = {1}.".format(numdevices, num_gpus))

    # Flag which gpus we can check
    if gpu_select is None:
        gpu_check = [True] * numdevices
    else:
        gpu_check = [False] * numdevices
        try:
            gpu_check[gpu_select] = True
        except TypeError:
            try:
                for i in gpu_select:
                    gpu_check[i] = True
            except Exception:
                raise GpuError("Please provide an int or an iterable of ints for 'gpu_select' parameter.")

    # Now check if any devices are suitable. Print out GPU device info. Useful for debugging.
    for i in range(numdevices):
        # If the gpu was specified, examine it
        if not gpu_check[i]:
            continue

        handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
        info = py3nvml.nvmlDeviceGetMemoryInfo(handle)

        # Sometimes GPU has a few MB used when it is actually free
        if (info.free + 10) / info.total >= gpu_fraction:
            gpu_free[i] = True
        else:
            logger.info('GPU {} has processes on it. Skipping.'.format(i))

        str_ = "GPU {}:\t".format(i) + \
               "Used Mem: {:>6}MB\t".format(info.used / (1024 ** 2)) + \
               "Total Mem: {:>6}MB".format(info.total / (1024 ** 2))
        logger.debug(str_)

    py3nvml.nvmlShutdown()

    # get available gpu numbers
    available_gpu = [i for i, x in enumerate(gpu_free) if x]
    if num_gpus > len(available_gpu):
        print("GpuWarning: only {0} of {1} gpu is available.".format(len(available_gpu), numdevices))
    else:
        available_gpu = available_gpu[0:num_gpus]

    return available_gpu
