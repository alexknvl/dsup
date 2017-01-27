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

sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'new'))

import fx
import easytime

import result_store


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


Tweet = recordtype('Tweet', 'sid attr arg1 arg2 text p time')


def read_predictions(time_range, lines):
    predictions = []
    wp_edits = {}
    attrs = set([])

    for line in lines:
        obj = json.loads(line)

        arg1 = obj['arg1']
        arg2 = obj['arg2']
        tweet = obj['tweet']
        p = obj['pred']

        sid = tweet['sid']
        text = tweet['words']
        time = obj['tweetDate']

        if time_range[0] <= time <= time_range[1]:
            predictions.append(Tweet(sid, None, arg1, arg2, text, p, time))

        for edit_str in obj['wpEdits']:
            edit = json.loads(edit_str)
            attrs.add(edit['key'])
            ep = tuple(sorted([arg1, arg2]))
            wp_edits.setdefault(ep, []).append(edit['timestamp'])

    attr = ','.join(attrs)
    for p in predictions:
        p.attr = attr

    return predictions, wp_edits


def read_predictions_gw(time_range, lines):
    predictions = []
    wp_edits = {}
    attrs = set([])

    for line in lines:
        obj = json.loads(line)

        arg1 = obj['arg1']
        arg2 = obj['arg2']
        tweet = obj['tweet']
        p = obj['pred']

        sid = tweet['sample_id'].replace('\t', '.')
        text = tweet['words']
        time = tweet['timestamp']

        if time_range[0] <= time <= time_range[1]:
            predictions.append(Tweet(sid, None, arg1, arg2, text, p, time))

        for edit_str in obj['wpEdits']:
            edit = json.loads(edit_str)
            attrs.add(edit['key'])
            ep = tuple(sorted([arg1, arg2]))
            wp_edits.setdefault(ep, []).append(edit['timestamp'])

    attr = ','.join(attrs)
    for p in predictions:
        p.attr = attr

    return predictions, wp_edits


def old_main():
    attrs = [s.split(',') for s in sys.argv[1].split(';')]
    base_dir = sys.argv[2]
    out_dir = sys.argv[3]

    TOP_TWEETS_SELECTED = 20000
    RESERVOIR_SIZE_PER_ENTITY = 5
    NEGATIVE_MULT = 0.1

    for attr_group in attrs:
        attr_group_name = ','.join(attr_group)
        normal_dir = '_'.join([
            attr_group_name, "normal", "lr", "1", "all", "0"])
        baseline_dir = '_'.join([
            attr_group_name, "baseline", "lr", "1", "all", "0"])

        normal_lines = open(os.path.join(base_dir, normal_dir, "predOut"))
        baseline_lines = open(os.path.join(base_dir, baseline_dir, "predOut"))

        normal_tweets, wp_edits = read_predictions(normal_lines)
        baseline_tweets, _ = read_predictions(baseline_lines)

        tweets = fx.concat(
            fx.slice(normal_tweets, TOP_TWEETS_SELECTED),
            fx.slice(baseline_tweets, TOP_TWEETS_SELECTED))

        tweets = fx.skip_duplicates(tweets, key=lambda s: s[0])

        ep_tweets = fx.group_by(tweets, key=lambda t: tuple(t[2:4]))
        for k, v in ep_tweets.iteritems():
            ep_tweets[k] = fx.reservoir_sample(v, RESERVOIR_SIZE_PER_ENTITY)
        tweets = list(fx.concat(*ep_tweets.values()))
        tweets.sort(key=lambda t: -t[-1])
        tweets = tweets[:500]

        print attr_group_name, len(tweets)

        N = int(len(tweets) * NEGATIVE_MULT)

        negatives = \
            fx.reservoir_sample(fx.slice(normal_tweets, TOP_TWEETS_SELECTED, None), N) + \
            fx.reservoir_sample(fx.slice(baseline_tweets, TOP_TWEETS_SELECTED, None), N)

        tweets = fx.shuffle(tweets + negatives)
        print len(tweets)

        # No batches.
        out = open(os.path.join(out_dir, attr_group_name), 'w+')
        for l in tweets:
            out.write('\t'.join(str(s) for s in l[:-1]) + '\n')
        out.close()

        # Batches.
        for n in xrange(len(tweets) / 100):
            batch_dir = os.path.join(out_dir, 'batch%s' % n)
            mkdir_p(batch_dir)

            out = open(os.path.join(batch_dir, attr_group_name), 'w+')
            for l in tweets[n*100:n*100+100]:
                out.write('\t'.join(str(s) for s in l[:-1]) + '\n')
            out.close()


def select_top(iterable, count, key, predicate, score=None, min_score=None):
    result = []
    discarded_keys = set()
    result_keys = set()
    done = False

    for v in iterable:
        k = key(v)
        if not predicate(v):
            discarded_keys.add(k)
            continue

        if k in discarded_keys:
            continue

        if k in result_keys:
            continue

        if score is not None and min_score is not None:
            if score(v) < min_score:
                result = [r for r in result if r[0] not in discarded_keys]
                break

        result.append((k, v))

        if count is not None and len(result) == count:
            result = [r for r in result if r[0] not in discarded_keys]
            if len(result) == count:
                break

    return [r[1] for r in result]


def select_top_fuzzy(iterable, at_least, at_most, metric, predicate,
                     score=None, min_score=None):
    result = []
    discarded = []

    for v in iterable:
        if not predicate(v):
            result = [r for r in result if metric(v, r) != 0]
            discarded.append(v)
            continue

        d = min(metric(v, r) for r in discarded) if len(discarded) > 0 else 1
        if d == 0:
            continue

        d = min(metric(v, r) for r in result) if len(result) > 0 else 1
        if d == 0:
            continue

        if score is not None and min_score is not None:
            if score(v) < min_score and \
                    (at_least is None or len(result) >= at_least):
                break

        result.append(v)

        if at_most is not None and len(result) >= at_most:
            break
    return result


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def ep_metric(ep1, ep2):
    return min(
        levenshtein(ep1[0], ep2[0]) + levenshtein(ep1[1], ep2[1]),
        levenshtein(ep1[0], ep2[1]) + levenshtein(ep1[1], ep2[0]))


def make_sticky_metric(metric, min_value):
    def result(a, b):
        r = metric(a, b)
        if r >= min_value:
            return r
        else:
            return 0
    return result


def new_main():
    time_range = (easytime.ts(year=2008, month=6, day=5),
                  easytime.ts(year=2012, month=1, day=1))

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
            predictions, wp_edits = read_predictions_gw(time_range, open(normal_path))
            systems = {
                'normal_lr': predictions,
                'baseline_lr': read_predictions_gw(time_range, open(baseline_path))[0]
            }
        else:
            predictions, wp_edits = read_predictions(time_range, open(normal_path))
            systems = {
                'normal_lr': predictions,
                'baseline_lr': read_predictions(time_range, open(baseline_path))[0]
            }

        month_bins = {}
        for name, tweets in systems.iteritems():
            for tweet in tweets:
                date = easytime.timestamp_to_datetime(tweet.time, 'utc')
                bin_id = (date.year, date.month)
                print bin_id
                lst = month_bins.setdefault(bin_id, {}).setdefault(name, [])
                lst.append(tweet)

        def check_edits(tweet):
            ep = tuple(sorted([tweet.arg1, tweet.arg2]))
            edits = wp_edits.get(ep, [])
            if len(edits) != 0:
                date = easytime.timestamp_to_datetime(tweet.time, 'utc')
                edit_date = easytime.timestamp_to_datetime(min(edits), 'utc')
                if (edit_date.year, edit_date.month) < (date.year, date.month):
                    return False
            return True

        def metric(a, b):
            r = ep_metric((a.arg1, a.arg2), (b.arg1, b.arg2))
            if r >= 3:
                return r
            else:
                return 0

        NON_W = re.compile('[^a-z]')

        def wkey(x):
            a1 = NON_W.sub('', x.arg1.lower())
            a2 = NON_W.sub('', x.arg2.lower())
            return a1 + '\t' + a2

        for date, bin in month_bins.iteritems():
            for name, tweets in bin.iteritems():
                # Select only the best result per entity pair.
                by_ep = fx.group_by(tweets, key=lambda t: (t.arg1, t.arg2))
                for k, v in by_ep.iteritems():
                    v = fx.max_by(v, key=lambda t: t.p)
                    by_ep[k] = v

                # Get the top 10 results
                top = select_top_fuzzy(
                    sorted(by_ep.itervalues(), key=lambda t: -t.p),
                    at_least=10, at_most=50,
                    metric=metric,
                    predicate=check_edits,
                    score=lambda t: t.p,
                    min_score=0.9 if name == 'baseline_lr' else 0.3)
                bin[name] = top

        out = open(os.path.join(out_dir, 'binned_' + attr_group_name), 'w+')
        for date, bin in month_bins.iteritems():
            for name, tweets in bin.iteritems():
                for t in tweets:
                    out.write(' '.join(str(s) for s in date) + '\t')
                    out.write(name + '\t')
                    out.write('\t'.join(str(s) for s in t) + '\n')
        out.close()

        # out = open(os.path.join(out_dir, attr_group_name), 'w+')
        # shuffled = fx.shuffle(t for date, bin in month_bins.iteritems()
        #                       for name, tweets in bin.iteritems()
        #                       for t in tweets)
        # for t in shuffled:
        #     out.write('\t'.join(str(s)
        #               for s in [t.sid, t.arg1, t.arg2, t.text]) + '\n')
        # out.close()

        print "Done with %s" % attr_group_name


def main3(args):
    time_range = (easytime.ts(year=2008, month=6, day=5),
                  easytime.ts(year=2012, month=1, day=1))

    dirs = [(path, attr, hash)
            for path, (attr, hash) in
            result_store.v3_get_all_output_dirs(args.inputdir)]

    for (path, attr, hash) in dirs:
        predictions = result_store.v2_read_predictions('test', path)

        month_bins = {}
        for p in predictions:
            date = easytime.timestamp_to_datetime(p.s.timestamp, 'utc')
            ym = (date.year, date.month)
            print ym

            month_bins\
                .setdefault(ym, {})\
                .setdefault(name, [])\
                .append(p.s)

        def check_edits(s):
            edits = wp_edits.get(s.args, [])
            if len(edits) != 0:
                date = easytime.timestamp_to_datetime(s.time, 'utc')
                edit_date = easytime.timestamp_to_datetime(min(edits), 'utc')
                if (edit_date.year, edit_date.month) < (date.year, date.month):
                    return False
            return True

        def metric(a, b):
            r = ep_metric((a.arg1, a.arg2), (b.arg1, b.arg2))
            if r >= 3:
                return r
            else:
                return 0

        NON_W = re.compile('[^a-z]')

        def wkey(x):
            a1 = NON_W.sub('', x.arg1.lower())
            a2 = NON_W.sub('', x.arg2.lower())
            return a1 + '\t' + a2

        for date, bin in month_bins.iteritems():
            for name, samples in bin.iteritems():
                # Select only the best result per entity pair.
                by_ep = fx.group_by(samples, key=lambda t: (t.arg1, t.arg2))
                for k, v in by_ep.iteritems():
                    v = fx.max_by(v, key=lambda t: t.p)
                    by_ep[k] = v

                # Get the top 10 results
                top = select_top_fuzzy(
                    sorted(by_ep.itervalues(), key=lambda t: -t.p),
                    at_least=10, at_most=50,
                    metric=metric,
                    predicate=check_edits,
                    score=lambda t: t.p,
                    min_score=0.9 if name == 'baseline_lr' else 0.3)
                bin[name] = top

        # out = open(os.path.join(out_dir, 'binned_' + attr_group_name), 'w+')
        # for date, bin in month_bins.iteritems():
        #     for name, tweets in bin.iteritems():
        #         for t in tweets:
        #             out.write(' '.join(str(s) for s in date) + '\t')
        #             out.write(name + '\t')
        #             out.write('\t'.join(str(s) for s in t) + '\n')
        # out.close()

        # out = open(os.path.join(out_dir, attr_group_name), 'w+')
        # shuffled = fx.shuffle(t for date, bin in month_bins.iteritems()
        #                       for name, tweets in bin.iteritems()
        #                       for t in tweets)
        # for t in shuffled:
        #     out.write('\t'.join(str(s)
        #               for s in [t.sid, t.arg1, t.arg2, t.text]) + '\n')
        # out.close()

        print "Done with %s" % attr_group_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdir', required=True)
    parser.add_argument('-o', '--outputdir', required=True)
    args = parser.parse_args()
    main3(args)

# if __name__ == "__main__":
#   predictions = []
#   attrs = set([])

#   for line in fileinput.input():
#     obj = json.loads(line)
#     tweet = obj['tweet']
#     sid = tweet['sid']
#     text = tweet['words']
#     arg1 = obj['arg1']
#     arg2 = obj['arg2']
#     predictions.append([sid, arg1, arg2, text])

#     for edit_str in obj['wpEdits']:
#       edit = json.loads(edit_str)
#       attrs.add(edit['key'])

#   attr = ','.join(attrs)

#   for p in predictions:
#     print '\t'.join(p[0:1] + [attr] + p[1:])
