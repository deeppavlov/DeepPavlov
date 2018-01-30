import argparse
from pathlib import Path
import sys

p = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(p))

from deeppavlov.core.data.utils import download, download_decompress
from deeppavlov.core.data.urls import REQ_URLS, ALL_URLS, EMBEDDING_URLS

parser = argparse.ArgumentParser()

parser.add_argument('-all', action='store_true',
                    help="Download everything. Warning! There should be at least 15 GB space"
                         " available on disk.")


def download_resources(args):
    if args.all:
        urls = ALL_URLS
    else:
        urls = REQ_URLS

    for url in urls:
        download_path = Path('../download')
        download_path.mkdir(exist_ok=True)
        dest_path = download_path

        embeddings_path = download_path.joinpath('embeddings')

        if url in EMBEDDING_URLS:
            embeddings_path.mkdir(exist_ok=True)
            dest_path = embeddings_path.joinpath(url.split("/")[-1])
            download(dest_path, url)

        else:
            download_decompress(url, dest_path)


def main():
    args = parser.parse_args()
    print("Downloading...")
    download_resources(args)
    print("\nDownload successful!")


if __name__ == "__main__":
    main()
