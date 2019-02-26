
__all__ = ['feb_common', 'nent_to_qent', 't1_parser', 't1_text_generator']


from question2wikidata.extractors import title
# print('Printed question2wikidata.extractors')
BOOK_NAMES_EXTRACTOR = title.titleExtractor()


LOG_FILE = open('LOG_FILE.jsonl', 'a', encoding = 'utf-8')
DEBUG = False