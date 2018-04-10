import os

WC_DATADIR = 'd:/data/indec'
QUORA_DATA_DIR = 'd:/data/quora'

QUORA_DATA_FILE = os.path.join(QUORA_DATA_DIR, 'quora-user-qa.json')

WC_DF_FILE = os.path.join(WC_DATADIR, 'docs-14k-words-df.txt')
WC_NAME_DOC_FILE = os.path.join(WC_DATADIR, 'name-doc.txt')
WC_DOC_CONTENT_FILE = os.path.join(WC_DATADIR, 'docs-14k-content.txt')
WC_SEG_DOC_CONTENT_FILE = os.path.join(WC_DATADIR, 'docs-14k-content-seg.txt')

QUORA_ANSWER_TOK_FILE = os.path.join(QUORA_DATA_DIR, 'answer-text-tok.txt')
QUORA_ANSWER_TOK_LOWER_FILE = os.path.join(QUORA_DATA_DIR, 'answer-text-tok-low.txt')
QUORA_DF_FILE = os.path.join(QUORA_DATA_DIR, 'quora-answers-df.txt')
QUORA_NER_NAME_CNT_FILE = os.path.join(QUORA_DATA_DIR, 'ner-name-cnts.txt')
QUORA_NAME_DOC_FILE = os.path.join(QUORA_DATA_DIR, 'name-doc.txt')
