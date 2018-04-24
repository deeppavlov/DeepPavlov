from typing import List, Any, Optional
import sqlite3

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.data.utils import download

DB_URL = 'http://lnsigo.mipt.ru/export/datasets/wikipedia/enwiki.db'


@register('wiki_sqlite_vocab')
class WikiSQLiteVocab(Component):
    """
    Get SQlite documents by ids.
    """

    def __init__(self, data_dir: str = '', data_url: str = DB_URL, **kwargs):
        """
        :param data_dir: a directory name where DB is located
        :param data_url: an URL to SQLite DB
        """
        download_dir = expand_path(data_dir)
        download_path = download_dir.joinpath(data_url.split("/")[-1])
        download(download_path, data_url, force_download=False)

        self.connect = sqlite3.connect(str(download_path), check_same_thread=False)
        self.db_name = self.get_db_name()

    def __call__(self, doc_ids: List[List[Any]], *args, **kwargs) -> List[str]:
        """
        Get the contents of files, stacked by space.
        :param questions: queries to search an answer for
        :param n: a number of documents to return
        :return: document contents (as a single string)
        """
        all_contents = []
        for ids in doc_ids:
            contents = [self.get_doc_content(doc_id) for doc_id in ids]
            contents = ' '.join(contents)
            all_contents.append(contents)
        return all_contents

    def get_db_name(self) -> str:
        cursor = self.connect.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        assert cursor.arraysize == 1
        name = cursor.fetchone()[0]
        cursor.close()
        return name

    def get_doc_content(self, doc_id: Any) -> Optional[str]:
        cursor = self.connect.cursor()
        cursor.execute(
            "SELECT text FROM {} WHERE id = ?".format(self.db_name),
            (doc_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]
