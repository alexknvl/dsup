from __future__ import division

import os
import errno
from unidecode import unidecode
import re
import numpy as np
import scipy.sparse

from collections import namedtuple
from recordtype import recordtype

import fx


def normalize_attr(attr):
    return ','.join(sorted(attr.split(',')))


Text = namedtuple('Text', ['words', 'pos', 'ner'])
Sample = recordtype('Sample', ['id', 'timestamp', 'text',
                               'args', 'edits', 'y', 'features'])
Edit = namedtuple('Edit', ['id', 'timestamp', 'relation', 'args'])
TPRF = namedtuple('TPRF', ['threshold', 'precision', 'recall', 'f1'])


def normalize_infobox(s):
    s = s.lower().replace(' ', '')
    if s.startswith('infobox'):
        s = s[len('infobox'):]
    if s.startswith('_'):
        s = s[1:]
    return s


def text_from_json(j):
    return Text(*j)


def edit_from_json(j):
    id, timestamp, relation, args = j
    args = tuple(args)
    return Edit(id, timestamp, relation, args)


def sample_from_json(j):
    id = j['id']
    timestamp = j['timestamp']
    text = text_from_json(j['text'])
    args = tuple(j['args'])
    edits = [edit_from_json(x) for x in j['edits']]
    y = j['y']
    features = j['features']
    return Sample(id, timestamp, text, args, edits, y, features)


def normalize_string(s, _whitespace_pattern=re.compile(r'\s+')):
    return _whitespace_pattern.sub(' ', unidecode(s).lower()).strip()


def sha1_structure(s):
    import hashlib
    import json
    return hashlib.sha1(json.dumps(s, sort_keys=True)).hexdigest()


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def gen_tprf(predictions, positive_count=None, shuffle=True, sort=True):
    predictions = predictions if not shuffle else \
        fx.shuffle(predictions)
    predictions = predictions if not sort else \
        sorted(predictions, key=lambda p: -p.p)

    N = positive_count if positive_count is not None else \
        sum(1 for x in predictions if x.y == 1)

    tp = 0
    fp = 0
    fn = 0

    for p in predictions:
        if p.y == 1:
            tp += 1
        elif p.y == -1:
            fp += 1
        # else:
        #     assert False

        fn = N - tp

        T = p.p
        P = 0.0
        if tp + fp > 0:
            P = tp / (tp + fp)
        R = 0.0
        if tp + fn > 0:
            R = tp / (tp + fn)
        F = 0.0
        if P + R > 0:
            F = 2 * P * R / (P + R)

        yield TPRF(
            threshold=T,
            precision=P,
            recall=R,
            f1=F)
