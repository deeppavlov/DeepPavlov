import argparse
import gzip
import sys
import tarfile
from hashlib import md5
from pathlib import Path
from typing import List, Dict, Union

from deeppavlov.core.data.utils import file_md5

parser = argparse.ArgumentParser()
parser.add_argument("fname", help="path to a file to compute hash for", type=str)
parser.add_argument('-o', '--outfile', help='where to write the hashes', default=None, type=str)


def tar_md5(fpath: Union[str, Path]) -> Dict[str, str]:
    tar = tarfile.open(fpath)
    res = {}
    while True:
        item: tarfile.TarInfo = tar.next()
        if item is None:
            break
        if not item.isfile():
            continue
        res[item.name] = md5(tar.extractfile(item).read()).hexdigest()
    return res


def gzip_md5(fpath: Union[str, Path], chunk_size: int = 2**16):
    file_hash = md5()
    with gzip.open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def main(args: List[str] = None) -> None:
    args = parser.parse_args(args)
    p = Path(args.fname).expanduser()
    if not p.is_file():
        raise RuntimeError(f'{p} is not a file')

    outfile = args.outfile
    if outfile is None:
        outfile = p.with_suffix(p.suffix + '.md5').open('w', encoding='utf-8')
    elif outfile == '-':
        outfile = sys.stdout
    else:
        outfile = Path(outfile).expanduser().open('w', encoding='utf-8')

    if '.tar' in {s.lower() for s in p.suffixes}:
        hashes = tar_md5(p)
    elif p.suffix.lower() == '.gz':
        hashes = {p.with_suffix('').name: gzip_md5(p)}
    else:
        hashes = {p.name: file_md5(p)}

    for fname, fhash in hashes.items():
        print(f'{fhash} *{fname}', file=outfile, flush=True)


if __name__ == '__main__':
    main()
