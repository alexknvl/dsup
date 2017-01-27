import sys, errno, os, os.path
import ujson as json
import gzip

from FeatureExtractor import *
from Classifier import *
from Vocab import *

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'new'))
import easytime as et, fx
from collections import namedtuple
from recordtype import recordtype

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def inside_range(range, x):
    return range[0] <= x and x <= range[1]

# Classifier: in_test_set, is_unknown, offset => 1 | 0 | -1
#   1 means positive
#   0 means "do not consider"
#  -1 means negative during testing and unknown/negative during training

POSITIVE_RANGE = [-10, 3]
NEGATIVE_RANGE = [-50, 3]
def timediff_classifier(in_test_set, is_unknown, tweet_offset):
    if is_unknown:
        return -1

    tweet_offset = tweet_offset / et.dt(days=1)

    if inside_range(POSITIVE_RANGE, tweet_offset):
        return 1
    elif not inside_range(NEGATIVE_RANGE, tweet_offset):
        return -1
    else:
        return 0

def baseline_classifier(in_test_set, is_unknown, tweet_offset):
    if is_unknown:
        return -1

    if in_test_set:
        return timediff_classifier(in_test_set, is_unknown, tweet_offset)
    else:
        return 1

CLASSIFIERS = {
    'normal': timediff_classifier,
    'baseline': baseline_classifier
}

Word  = namedtuple('Word',  ['text', 'pos', 'ne'])
Tweet = namedtuple('Tweet', ['id', 'timestamp', 'words'])
Edit  = namedtuple('Edit',  ['id', 'timestamp'])

def compute_prf(predictions, total_positive = None):
    """ Computes a single precision and recall of a list of predictions """
    tp = 0.0
    fp = 0.0
    fn = 0.0

    if total_positive is None:
        total_positive = sum(1 for p in predictions if p['y'] == 1)

    for p in predictions:
        T = p['pred']
        if p['y'] == 1:
            tp += 1
        elif p['y'] == -1:
            fp += 1

        fn = total_positive - tp
        P = tp / (tp + fp) if tp + fp > 0 else 0
        R = tp / (tp + fn) if tp + fn > 0 else 0
        F = 2 * P * R / (P + R) if P + R > 0 else 0
        yield (T, P, R, F)

def json_edit_to_internal(json):
    id = int(json['id'])
    ts = int(json['timestamp'])
    return Edit(id, ts)

def json_tweet_to_internal(json):
    id    = int(json['sid'])
    timestamp = et.strptime(json['created_at'], '%a %b %d %H:%M:%S +0000 %Y', 'utc')
    text  = json['words'].split(' ')
    pos   = json['pos'].split(' ')
    ne    = [x[0] for x in json['neTags'].split(' ')]

    return Tweet(id, timestamp, map(Word, text, pos, ne))

def parse_positive_lines(lines):
    for line in lines:
        (label, rel, arg1, arg2,
         tweetDate, editDate, timeSinceTweet,
         tweetStr, editStr) = line.strip().split('\t')

        edit = json_edit_to_internal(json.loads(editStr))
        tweet = json_tweet_to_internal(json.loads(tweetStr))

        yield (arg1, arg2, tweet, edit)

def parse_negative_lines(lines):
    for line in lines:
        (arg1, arg2, tweetDate, tweetStr) = line.strip().split('\t')
        tweet = json_tweet_to_internal(json.loads(tweetStr))
        yield (arg1, arg2, tweet)

Sample  = namedtuple('Sample', ['arg1', 'arg2', 'tweet', 'is_unknown', 'offset'])
Window  = recordtype('Window', ['arg1', 'arg2', 'tweet', 'x', 'y'])
DataSet = namedtuple('DataSet', ['samples', 'edits_by_ep', 'samples_by_ep'])

MIN_FEATURE_COUNT   = 3
MIN_TWEET_COUNT     = 5
MODES               = ["normal", "baseline"]
XR_MODES            = ["lrxr", "lr"]
FEATURE_WINDOW_DAYS = [1]
EXPECTATIONS        = [0.25]
TRAIN_RANGE = (et.ts(year=2008, month=9, day=1),
               et.ts(year=2011, month=6, day=1))
DEV_RANGE   = (et.ts(year=2011, month=6, day=5),
               et.ts(year=2012, month=1, day=1))
NEGATIVE_FILE = "../training_data2/negative_small.gz"

def read_data(attributes):
    positive_files = [iter(open(os.path.join('../training_data2', attr)))
                      for attr in attributes]
    positive_lines = fx.concat(*positive_files)
    parsed_pos_lines = fx.skip_duplicates(parse_positive_lines(positive_lines),
                                          key=lambda t: str(t[2].id) + '\t' +
                                                        t[0] + '\t' + t[1])

    train = DataSet([], [], {}, {})
    test  = DataSet([], [], {}, {})
    known_ids = set([])

    for arg1, arg2, tweet, edit in parsed_pos_lines:
        in_train_range = inside_range(TRAIN_RANGE, tweet.timestamp)
        in_dev_range   = inside_range(DEV_RANGE, tweet.timestamp)

        if not in_train_range and not in_dev_range:
            continue

        dataset = train if in_train_range else test

        offset = tweet.timestamp - edit.timestamp
        sample = Sample(arg1, arg2, tweet, is_unknown=False, offset)
        dataset.samples.append(sample)
        dataset.edits_by_ep.setdefault((arg1, arg2), []).append(edit)
        dataset.samples_by_ep.setdefault((arg1, arg2), []).append(sample)

        known_ids.add(tweet.id)

    print "len(train.samples)=%s" % len(train.samples)
    print "len(test.samples)=%s" % len(test.samples)

    negative_lines = iter(gzip.open(NEGATIVE_FILE))
    parsed_neg_lines = fx.skip_duplicates(parse_negative_lines(negative_lines),
                                          key=lambda t: str(t[2].id) + '\t' +
                                                        t[0] + '\t' + t[1])

    negative_count = 0
    for arg1, arg2, tweet in parsed_neg_lines:
        negative_count += 1
        if negative_count % 100000 == 0:
            print "number of negative read: %s" % negative_count

        if tweet.id in known_ids:
            continue

        in_train_range = inside_range(TRAIN_RANGE, tweet.timestamp)
        in_dev_range   = inside_range(DEV_RANGE, tweet.timestamp)

        if not in_train_range and not in_dev_range:
            continue

        dataset = train if in_train_range else test

        sample = Sample(arg1, arg2, tweet, is_unknown=True, float('nan'))
        dataset.samples.append(sample)
        dataset.edits_by_ep.setdefault((arg1, arg2), []).append(edit)
        dataset.samples_by_ep.setdefault((arg1, arg2), []).append(sample)

    sample_timestamp_cmp = \
        lambda a,b: cmp(a.tweet.timestamp, b.tweet.timestamp)
    train.samples.sort(sample_timestamp_cmp)
    test.samples.sort(sample_timestamp_cmp)

    print "len(train.samples)=%s" % len(train.samples)
    print "len(test.samples)=%s" % len(test.samples)

    return (train, test)

def filter_features_by_entity_count(windows, min_count):
    feature_counts = {}

    for w in windows:
        for f in w.x:
            feature_counts.setdefault(f, set()).add((w.arg1, w.arg2))

    for f, eps in feature_counts.iteritems():
        feature_counts[f] = len(eps)

    for w in windows:
        w.x = filter(lambda f: feature_counts[f] >= min_count, w.x)

def extract_features(self, train, test, feature_window, mode):
    print "extract_features"

    train_windows = list(generate_data(train.samples,
                                       train.samples_by_ep,
                                       feature_window))
    test_windows  = list(generate_data(test.samples,
                                       test.samples_by_ep,
                                       feature_window))

    filter_features_by_entity_count(train_windows, MIN_FEATURE_COUNT)

    print "len(train_windows)=%s" % len(train_windows)
    print "len(test_windows)=%s" % len(test_windows)
    return (train_windows, test_windows)

def generate_ne_indices(words):
    start = -1

    for index, word in enumerate(words):
        if word.ne == 'B':
            if start != -1:
                yield (start, index)
            start = index
        elif word.ne == 'O' and start != -1:
            yield (start, index)
            start = -1

    if start != -1:
        yield (start, index + 1)

def entity_indices(words, entities):
    result = [[] for e in entities]

    for start, end in generate_ne_indices(words):
        segment = ' '.join(x.text.lower() for x in words[start:end])
        for i, entity in enumerate(entities):
            if segment == entity:
                result[i].append((start, end))
    return result

def generate_word_features_between(e1, e2, context_size, words,
                                   max_words_between):
    left_context = ' '.join(words[e1[1]-context_size:e1[1]])
    right_context = ' '.join(words[e2[2]:e2[2]+context_size])
    between_context = ' '.join(words[e1[2]:e2[1]])

    between_context_short = between_context
    if e2[1] - e1[2] > 2 * context_size:
        between_context_short = \
            ' '.join(words[e1[2]:e1[2]+context_size]) + \
            ' ... ' + \
            ' '.join(words[e2[1]-context_size:e2[1]])

    if e2[1] - e1[2] <= max_words_between:
        yield ' '.join((e1[0], between_context, e2[0]))
        yield ' '.join((left_context, e1[0], between_context, e2[0]))
        yield ' '.join((e1[0], between_context, e2[0], right_context))

    yield ' '.join((e1[0], between_context_short, e2[0]))
    yield ' '.join((left_context, e1[0], between_context_short, e2[0]))
    yield ' '.join((e1[0], between_context_short, e2[0], right_context))

def generate_pos_features_between(e1, e2, pos_tags, max_pos_between):
    between = ' '.join(pos_tags[e1[2]:e2[1]])
    if e2[1] - e1[2] <= max_pos_between:
        yield ' '.join((e1[0], between, e2[0]))


def merge_nv_pos(words):
    result = []
    for word in words:
        if word.pos[0] == 'N' or word.pos[0] == 'V':
            result.append(word.text)
        else:
            result.append(word.pos)
    return result

def generate_nv_features_between(e1, e2, merged_tokens, max_nv_between):
    if e2[1] - e1[2] <= max_nv_between:
        left_context = merged_tokens[e1[1]-2:e1[1]]
        between_context = merged_tokens[e1[2]:e2[1]]
        right_context = merged_tokens[e2[2]:e2[2]+2]
        yield ' '.join((e1[0], between_context, e2[0]))
        yield ' '.join((left_context, e1[0], between_context, e2[0]))
        yield ' '.join((e1[0], between_context, e2[0], right_context))

def generate_lexical_features(arg1, arg2, words,
                              max_words_between=5,
                              max_pos_between=8,
                              max_nv_between=8):

    a1indices, a2indices = entity_indices(words, [arg1, arg2])
    merged_tokens = merge_nv_pos(words)
    plain_words = [w.text for w in words]
    plain_pos   = [w.pos for w in words]

    for a1 in a1indices:
        for a2 in a2indices:
            if a1[0] == a2[0]:
                continue

            if a2[0] - a1[1] > 0:
                e1 = ('arg1', a1[0], a1[1])
                e2 = ('arg2', a2[0], a2[0])
            elif a1[0] - a2[1] > 0:
                e1 = ('arg2', a2[0], a2[1])
                e2 = ('arg1', a1[0], a1[0])
            else:
                continue

            # Words between (e.g. u'arg2 secretly wed arg1').
            for context_size in [1, 2, 3]:
                for f in generate_word_features_between(
                        e1, e2, context_size, plain_words, max_words_between):
                    yield f

            # POS between (e.g. u'arg1 VBN TO NNP arg2').
            for f in generate_pos_features_between(
                    e1, e2, plain_pos, max_pos_between):
                yield f

            # Nouns + verb between (e.g. u'arg2 ADV wed arg1').
            for f in generate_nv_features_between(
                    e1, e2, merged_tokens, max_nv_between):
                yield f

def generate_data(samples, samples_by_ep, feature_window_days):
    for s in fx.throttle_threads(samples,
                                 thread_key=lambda s: (s.arg1, s.arg2),
                                 time_key=lambda s: (s.tweet.timestamp),
                                 delay=et.dt(minutes=10)):
        all_samples = samples_by_ep.get((s.arg1, s.arg2), [])

        if len(all_samples) >= MIN_TWEET_COUNT:
            predicate = lambda x: x.tweet.timestamp <= s.tweet.timestamp and \
                                  x.tweet.timestamp > (s.tweet.timestamp - feature_window)
            window = (x.tweet for x in all_samples if predicate(x))

            fx.concat(generate_lexical_features(arg1, arg2, x.words) for x in window)
            yield Window(arg1, arg2, target, features, False)

def MakePredictions(self, model, devData):
    for fields in devData:
        x = fields[0]
        y = fields[1]
        prediction = model.Predict(x)
        #tweetDate = fields[2]
        tweet = fields[5]
        tweet['datetime'] = parse_tweet_date(tweet['created_at'])
        tweetDate = tweet['created_at']
        arg1 = fields[3]
        arg2 = fields[4]
        ep = "%s\t%s" % (arg1,arg2)
        #print ep
        wpEdits = list(self.wpEdits.get(ep, []))
        editDate = None
        if len(wpEdits) > 0:
            editDate = datetime.fromtimestamp(json.loads(wpEdits[0])['timestamp'])

        yield {'y':y, 'pred':prediction,
               'arg1':arg1, 'arg2':arg2,
               'tweetDate':tweetDate, 'editDate':editDate,
               'tweet':tweet , 'wpEdits':wpEdits}

def main(attributes, output_dir):
    train, test = read_data(attributes)

    for nsr, fwd in fx.product(NEGATIVE_SAMPLE_RATE, self.FEATURE_WINDOW_DAYS):
        train_data, dev_data = extract_features(train, test, fwd)

        for mode, xr in fx.product(self.MODES, self.XR_MODES):
            print "Reclassifying"
            self.ClassifyAll(mode)
            self.UpdateDataClasses(trainData, devData)

            model = LR_XRclassifierV2() if xr == "lrxr" else LRclassifierV2()
            model.Prepare([x for x in trainData if x[1] != 0])
            print "n features: %s" % model.vocab.GetVocabSize()

            for p_ex in self.EXPECTATIONS if xr == "lrxr" else [0]:
                if xr == "lrxr":
                    model.Train(p_ex=p_ex, l2=100.0)
                else:
                    model.Train(l2=100.0)

                predictions = self.MakePredictions(model, devData)

                N = sum(1 for x in predictions if x['y'] == 1)
                predictions.sort(lambda a,b: cmp(b['pred'], a['pred']))
                F = self.MaxF1(predictions, N)

if __name__ == "__main__":
    ds = GridSearch(sys.argv[1].split(','))
    ds.Run(sys.argv[2])
