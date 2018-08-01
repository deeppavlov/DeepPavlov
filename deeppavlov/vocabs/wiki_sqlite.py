from typing import List, Any, Optional

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger

from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

logger = get_logger(__name__)


@register('wiki_sqlite_vocab')
class WikiSQLiteVocab(SQLiteDataIterator):
    """Get content from SQLite database by document ids.

    Args:
        data_url: an URL where to download a DB from
        data_dir:  a directory where to save downloaded DB to

    """

    def __init__(self, data_url, data_dir: str = '', **kwargs):

        super().__init__(data_dir=data_dir, data_url=data_url)

    def __call__(self, doc_ids: Optional[List[List[Any]]] = None, *args, **kwargs) -> List[str]:
        """Get the contents of files, stacked by space.

        Args:
            doc_ids: a batch of lists of ids to get contents for

        Returns:
            a list of contents
        """
        all_contents = []
        if not doc_ids:
            logger.warn('No doc_ids are provided in WikiSqliteVocab, return all docs')
            doc_ids = [self.get_doc_ids()]

        for ids in doc_ids:
            contents = [self.get_doc_content(doc_id) for doc_id in ids]
            contents = ' '.join(contents)
            all_contents.append(contents)

        return all_contents
