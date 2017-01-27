from __future__ import division
import re
import ujson as json
import sys
from collections import namedtuple
import fx
import os
import gzip

TPRF = namedtuple('TPRF', ['threshold', 'precision', 'recall', 'f1'])
Prediction = namedtuple('Prediction', ['p', 'y', 'yes', 'no', 'ns', 'hy'])


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


def normalize_attr(attr):
    return ','.join(sorted(attr.split(',')))


def parse_old_result(line, results):
    (old_id, sid, attr, arg1, arg2, text,
        yes, no, ns) = line.rstrip('\r\n').split('\t')
    yes, no, ns = int(yes), int(no), int(ns)
    attr = normalize_attr(attr)
    results.setdefault(sid, {})\
        .setdefault((attr, arg1, arg2), [])\
        .append((yes, no, ns))


def parse_new_result(line, results):
    (year_month, mode, sid, attr, arg1, arg2, text, score, timestamp, mid,
        yes, no, ns) = line.rstrip('\n').split('\t')
    yes, no, ns = int(yes), int(no), int(ns)
    attr = normalize_attr(attr)
    try:
        int(sid)
    except:
        return
    results.setdefault(sid, {})\
        .setdefault((attr, arg1, arg2), [])\
        .append((yes, no, ns))


def parse_expert_annotated(line, results):
    (score, attr, mode, date, p, y,
     id, arg1, arg2, text) = line.rstrip('\n').split('\t')

    yes, no, ns = 0, 0, 0
    if score == '+':
        yes += 1
    elif score == '-':
        no += 1
    elif score in ['?', '?a', ]:
        ns += 1
    else:
        return

    results.setdefault(id, {})\
        .setdefault((attr, arg1, arg2), [])\
        .append((yes, no, ns))

def parse_prediction(line):
    p, line = line.split('\t')
    p = float(p)
    j = json.loads(line)
    y = j['y']
    arg1, arg2 = j['args']
    sid = j['id']
    return (sid, arg1, arg2, y, p)

if __name__ == '__main__':
    RESULTS = {}

    # for line in open('/home/konovalo/dsup_event/new-data/mturk-old-results.csv'):
    #     parse_old_result(line, RESULTS)
    #
    # for line in open('/home/konovalo/dsup_event/new-data/mturk-new-results.csv'):
    #     parse_new_result(line, RESULTS)

    for f in os.listdir('./full_run_turk_anno'):
        for line in open(os.path.join('./full_run_turk_anno', f)):
            parse_expert_annotated(line, RESULTS)

    KNOWN_SIDS = set(RESULTS.keys())

    ATTR = normalize_attr(sys.argv[1])
    PREDICTIONS = []

    for line in gzip.open(sys.argv[2]):
        sid, arg1, arg2, hy, p = parse_prediction(line)
        if sid not in KNOWN_SIDS:
            continue
        if (ATTR, arg1, arg2) not in RESULTS[sid]:
            continue
        yes, no, ns = 0, 0, 0
        for y, n, u in RESULTS[sid][(ATTR, arg1, arg2)]:
            yes += y
            no += n
            ns += u

        PREDICTIONS.append(Prediction(
            p=p, y=1 if yes > no else -1,
            yes=yes, no=no, ns=ns,
            hy=hy))

    max_f1 = 0
    for (t, p, r, f) in gen_tprf(PREDICTIONS, sort=True, shuffle=True):
        print(t, p, r, f)
        if max_f1 < f:
            max_f1 = f
    print max_f1
