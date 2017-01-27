import gzip
import ujson as json
from unidecode import unidecode

import easytime

from common import Text, Sample, Edit, normalize_string, normalize_infobox


def ne_tags_to_bio(tags):
    result = ['O']
    for t in tags:
        if t != 'O':
            if result[-1] == 'O':
                result.append('B')
            else:
                result.append('I')
        else:
            result.append('O')

    #assert len(result) == len(tags) + 1

    return result[1:]


def parse_matched_line(line):
    rel, arg1, arg2, edit_str, sentence_str = \
        line.decode('utf-8').strip().split('\t')
    edit_json = json.loads(edit_str)
    sentence_json = json.loads(sentence_str)

    doc_id = sentence_json['doc_id']
    sentence_id = sentence_json['sentence'][0]
    sid = "%s\t%s" % (doc_id, sentence_id)
    timestamp = easytime.ts(*sentence_json['date'])

    arg1 = normalize_string(arg1)
    arg2 = normalize_string(arg2)
    infobox = normalize_infobox(edit_json['infobox_name'])

    sentence = sentence_json['sentence'][1]
    text = Text(
        words=[normalize_string(text).replace(' ', '')
               for i, text, lemma, s, e, pos, ner in sentence],
        pos=[pos for i, text, lemma, s, e, pos, ner in sentence],
        ner=ne_tags_to_bio(ner for i, text, lemma, s, e, pos, ner in sentence))

    edit = Edit(id=edit_json['id'], timestamp=edit_json['timestamp'],
                relation=infobox + '/' + rel, args=[arg1, arg2])

    return Sample(id=sid, timestamp=timestamp, text=text,
                  args=(arg1, arg2), edits=[edit], y=-1, features=None)


def read_matched(path):
    with open(path) as input_file:
        for line in input_file:
            yield parse_matched_line(line)


def parse_unmatched_line(line):
    try:
        _, arg1, arg2, ne1, ne2, sentence_str = \
            line.decode('utf-8').strip().split('\t')
    except:
        return None

    sentence_json = json.loads(sentence_str)

    doc_id = sentence_json['doc_id']
    sentence_id = sentence_json['sentence'][0]
    sid = "%s\t%s" % (doc_id, sentence_id)
    timestamp = easytime.ts(*sentence_json['date'])

    arg1 = unidecode(arg1).lower()
    arg2 = unidecode(arg2).lower()

    sentence = sentence_json['sentence'][1]
    text = Text(
        words=[normalize_string(text).replace(' ', '')
               for i, text, lemma, s, e, pos, ner in sentence],
        pos=[pos for i, text, lemma, s, e, pos, ner in sentence],
        ner=ne_tags_to_bio(ner for i, text, lemma, s, e, pos, ner in sentence))

    return Sample(id=sid, timestamp=timestamp, text=text,
                  args=(arg1, arg2), edits=[], y=-1, features=None)


def read_unmatched(path):
    with gzip.open(path) as input_file:
        for line in input_file:
            yield parse_unmatched_line(line)
