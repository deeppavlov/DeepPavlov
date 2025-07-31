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

import argparse
import logging
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:
    raise ModuleNotFoundError(f"Please, run `pip install nbformat==5.8.0` before using this script.")

logging.basicConfig(level=logging.INFO, format="%(message)s")


def merge_markdown(nb: nbf.notebooknode.NotebookNode) -> None:
    """Merges consequent markdown cells into one."""
    start_idx = None
    slices = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                if i - start_idx > 1:
                    slices.append(slice(start_idx, i))
                start_idx = None
    for sl in slices[::-1]:
        nb["cells"][sl.start]["source"] = "\n\n".join([c["source"].rstrip() for c in nb["cells"][sl]])
        del nb["cells"][sl.start + 1: sl.stop]  # nb["cells"][sl] does not work properly


def drop_metadata(nb: nbf.notebooknode.NotebookNode) -> None:
    """Replaces notebook and cells metadata with empty dicts."""
    nb["metadata"] = dict()
    for i in range(len(nb["cells"])):
        nb["cells"][i]["metadata"] = dict()


def update_file(path: Path, update_ckpts: bool) -> None:
    """Optimizes ipynb files in order to reduce further git diffs.
    Args:
        path: File to update, if this is file. If this is dir - recursively searches and updates .ipynb files in it.
        update_ckpts: If False and path is dir, will skip all found ipynb files from .ipynb_checkpoints.
    """
    if path.is_dir():
        logging.info(f"Updating .ipynb files in {path} dir"
                     f"{', excluding files from .ipynb_checkpoints subdirs' if update_ckpts is False else ''}.")
        for f in path.rglob('*.ipynb'):
            if update_ckpts is False and '.ipynb_checkpoints' in f.parts:
                continue
            update_file(f, update_ckpts)
    else:
        logging.info(f"Updating {path}.")
        nb = nbf.read(path, nbf.NO_CONVERT)
        merge_markdown(nb)
        drop_metadata(nb)
        with open(path, "w") as fout:
            nbf.write(nb, fout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", help="path to an ipynb file to optimize", type=Path)
    parser.add_argument("--update-ckpts", help="update checkpoints in .ipynb_checkpoints subdirs", action="store_true")
    args = parser.parse_args()
    update_file(args.fname.resolve(), args.update_ckpts)


if __name__ == "__main__":
    main()
