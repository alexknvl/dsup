import sys
import os
import os.path
import errno

import re
import ujson as json
import gc
import fileinput

from recordtype import recordtype
from collections import namedtuple

import fx
import easytime

Prediction = recordtype(
    'Prediction', 'sid attr arg1 arg2 text p time y offset date')


def read_predictions(attr_group_name, lines):
    predictions = []
    wp_edits = {}

    for line in lines:
        obj = json.loads(line)

        arg1 = obj['arg1']
        arg2 = obj['arg2']
        tweet = obj['tweet']
        p = obj['pred']

        sid = tweet['sid']
        text = tweet['words']
        time = obj['tweetDate']

        randomly_sampled = tweet['from_negative']
        y = obj['y'] if not randomly_sampled else 0
        offset = -tweet['time_since_tweet']

        date = easytime.timestamp_to_datetime(time, 'utc')
        date = (date.year, date.month)

        predictions.append(Prediction(
            sid, attr_group_name, arg1, arg2, text, p, time, y, offset, date))

        for edit_str in obj['wpEdits']:
            edit = json.loads(edit_str)
            ep = tuple(sorted([arg1, arg2]))
            wp_edits.setdefault(ep, []).append(edit['timestamp'])

    return predictions, wp_edits


def read_predictions_gw(attr_group_name, lines):
    predictions = []
    wp_edits = {}

    for line in lines:
        obj = json.loads(line)

        arg1 = obj['arg1']
        arg2 = obj['arg2']
        tweet = obj['tweet']
        p = obj['pred']

        sid = tweet['sample_id'].replace('\t', '.')
        text = tweet['words']
        time = tweet['timestamp']

        randomly_sampled = tweet['randomly_sampled']
        y = obj['y'] if not randomly_sampled else 0
        offset = -tweet['dt']

        date = easytime.timestamp_to_datetime(time, 'utc')
        date = (date.year, date.month)

        predictions.append(Prediction(
            sid, attr_group_name, arg1, arg2, text, p, time, y, offset, date))

        for edit_str in obj['wpEdits']:
            edit = json.loads(edit_str)
            ep = tuple(sorted([arg1, arg2]))
            wp_edits.setdefault(ep, []).append(edit['timestamp'])

    return predictions, wp_edits


def select_top(iterable, key):
    result = []
    result_keys = set()

    for v in iterable:
        k = key(v)

        if k in result_keys:
            continue

        result.append(v)
        result_keys.add(k)

    return result


if __name__ == "__main__":
    attrs = [s.split(',') for s in sys.argv[1].split(';')]
    base_dir = sys.argv[2]
    out_dir = sys.argv[3]
    mode = sys.argv[4]

    for attr_group in attrs:
        attr_group_name = ','.join(attr_group)

        normal_path = os.path.join(
            base_dir,
            '_'.join([attr_group_name, "normal", "lr", "1", "all", "0"]),
            'predOut')
        baseline_path = os.path.join(
            base_dir,
            '_'.join([attr_group_name, "baseline", "lr", "1", "all", "0"]),
            'predOut')

        if mode == 'gw':
            predictions, wp_edits = read_predictions_gw(attr_group_name, open(normal_path))
            systems = {
                'normal_lr': predictions,
                'baseline_lr': read_predictions_gw(attr_group_name, open(baseline_path))[0]
            }
        else:
            predictions, wp_edits = read_predictions(attr_group_name, open(normal_path))
            systems = {
                'normal_lr': predictions,
                'baseline_lr': read_predictions(attr_group_name, open(baseline_path))[0]
            }

        def check_edits(tweet):
            ep = tuple(sorted([tweet.arg1, tweet.arg2]))
            edits = wp_edits.get(ep, [])
            if len(edits) != 0:
                date = easytime.timestamp_to_datetime(tweet.time, 'utc')
                edit_date = easytime.timestamp_to_datetime(min(edits), 'utc')
                if (edit_date.year, edit_date.month) < (date.year, date.month):
                    return False
            return True

        NON_W = re.compile('[^a-z]')

        def wkey(x):
            a1 = NON_W.sub('', x.arg1.lower())
            a2 = NON_W.sub('', x.arg2.lower())
            return a1 + '\t' + a2

        for name, tweets in systems.iteritems():
            # Select only the best result per entity pair.
            by_ep = fx.group_by(tweets, key=lambda t: (t.arg1, t.arg2))
            for k, v in by_ep.iteritems():
                v = fx.max_by(v, key=lambda t: t.p)
                by_ep[k] = v

            # Get the top 10 results
            top = select_top(
                sorted(by_ep.itervalues(), key=lambda t: -t.p),
                key=wkey)
            systems[name] = top

        # Prediction = recordtype(
        #     'Prediction', 'sid attr arg1 arg2 text p time y offset date')

        # year_month
        # baseline_lr/normal_lr
        # tweetid/gigawordid
        # attribute e1 e2 text
        # score timestamp
        # heuristic_alignment_score (-1/1 if aligned/unaligned, 0 if unlabeled)
        # nearest_edit_offset (unlabeled have this field set to 0,
        #                      otherwise edit_timestamp - tweet_timestamp)

        out = open(os.path.join(out_dir, 'full_' + attr_group_name), 'w+')
        for name, tweets in systems.iteritems():
            for t in tweets:
                out.write(' '.join(str(x) for x in t.date) + '\t')
                out.write(name + '\t')
                out.write('\t'.join(str(x) for x in [
                    t.sid, t.attr, t.arg1, t.arg2, t.text,
                    t.p, t.time, t.y, t.offset]))
                out.write('\n')
        out.close()


