import os

WC_DATADIR = 'd:/data/indec'
QUORA_DATA_DIR = 'd:/data/quora'
# QUORA_DATA_DIR = '/home/data/hldai/quora'

QUORA_DATA_FILE = os.path.join(QUORA_DATA_DIR, 'quora-user-qa.json')

COMMON_CH_WORDS_FILE = os.path.join(WC_DATADIR, 'common-words.txt')

WC_DF_FILE = os.path.join(WC_DATADIR, 'docs-14k-words-df.txt')
WC_NAME_DOC_FILE = os.path.join(WC_DATADIR, 'name-doc.txt')
WC_DOC_CONTENT_FILE = os.path.join(WC_DATADIR, 'docs-14k-content.txt')
WC_SEG_DOC_CONTENT_FILE = os.path.join(WC_DATADIR, 'docs-14k-content-seg.txt')

WC_DOC_INFO_NODUP_FILE = os.path.join(WC_DATADIR, 'docs-14k-nodup.txt')
WC_DOC_CONTENT_NODUP_FILE = os.path.join(WC_DATADIR, 'docs-14k-content-nodup.txt')
WC_SEG_DOC_CONTENT_NODUP_FILE = os.path.join(WC_DATADIR, 'docs-14k-content-seg-nodup.txt')
WC_DF_ND_FILE = os.path.join(WC_DATADIR, 'docs-14k-words-df.txt')
WC_NAME_DOC_ND_FILE = os.path.join(WC_DATADIR, 'name-doc-nd.txt')
WC_ENTITY_NAMES_FILE = os.path.join(WC_DATADIR, 'entity-names.txt')
WC_SEG_SENT_FILE = os.path.join(WC_DATADIR, 'docs-14k-sents-seg.txt')
WC_SENT_FILE = os.path.join(WC_DATADIR, 'docs-14k-sents.txt')
WC_MINIDOC_INFO_FILE = os.path.join(WC_DATADIR, 'docs-14k-minidocs-info.txt')
WC_MINIDOC_TEXT_FILE = os.path.join(WC_DATADIR, 'docs-14k-minidocs-text.txt')
WC_MINIDOC_TEXT_SEG_FILE = os.path.join(WC_DATADIR, 'docs-14k-minidocs-text-seg.txt')
WC_MINIDOC_INFO_NODUP_FILE = os.path.join(WC_DATADIR, 'docs-14k-minidocs-info-nodup.csv')
WC_MINIDOC_TEXT_NODUP_FILE = os.path.join(WC_DATADIR, 'docs-14k-minidocs-text-nodup.txt')
WC_MINIDOC_TEXT_SEG_NODUP_FILE = os.path.join(WC_DATADIR, 'docs-14k-minidocs-text-seg-nodup.txt')


QUORA_ANSWER_CONTENT_FILE = os.path.join(QUORA_DATA_DIR, 'answer-text.txt')
QUORA_ANSWER_NER_FILE = os.path.join(QUORA_DATA_DIR, 'answer-text-ner.txt')
QUORA_ANSWER_TOK_FILE = os.path.join(QUORA_DATA_DIR, 'answer-text-tok.txt')
QUORA_ANSWER_TOK_LOWER_FILE = os.path.join(QUORA_DATA_DIR, 'answer-text-tok-low.txt')
QUORA_DF_FILE = os.path.join(QUORA_DATA_DIR, 'quora-answers-df.txt')
QUORA_NER_NAME_CNT_FILE = os.path.join(QUORA_DATA_DIR, 'ner-name-cnts.txt')
QUORA_NAME_DOC_FILE = os.path.join(QUORA_DATA_DIR, 'name-doc.txt')
QUORA_NUM_TOTAL_DOCS = 143479
