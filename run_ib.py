import os.path
import gzip

import multiprocessing as mp
import ujson as json
import numpy as np
import scipy.sparse
import numpy.random

from tabulate import tabulate
from collections import namedtuple, frozenset, frozendict
from recordtype import recordtype

import easytime
import fx
import lrxr

ONE_DAY = easytime.dt(days=1)


def gen_tprf(predictions, n=None, sort=False):
    """ Computes precision, recall, f1 for a list of predictions."""
    if n is None:
        n = sum(1 for x in predictions if x['y'] == 1)

    if sort:
        predictions = sorted(predictions, key=lambda x: -x['pred'])

    tp = 0.0
    fp = 0.0
    fn = 0.0

    for p in predictions:
        T = p['pred']
        if p['y'] == 1:
            tp += 1
        elif p['y'] == -1:
            fp += 1

        fn = n - tp

        P = 0
        if tp + fp > 0:
            P = tp / (tp + fp)
        R = 0
        if tp + fn > 0:
            R = tp / (tp + fn)
        F = 0
        if P + R > 0:
            F = 2 * P * R / (P + R)

        yield (T, P, R, F)


def make_vocabularies(sample_features):
    vocabulary = {}
    reverse_vocabulary = []
    for features in sample_features:
        for f in features.iterkeys():
            if f not in vocabulary:
                vocabulary[f] = len(reverse_vocabulary)
                reverse_vocabulary.append(f)
    return vocabulary, reverse_vocabulary


def feature_matrix(sample_features, vocabulary, total_features):
    x = scipy.sparse.lil_matrix((len(sample_features), total_features))
    for i, fd in enumerate(sample_features):
        for f, v in fd.iteritems():
            j = vocabulary.get(f, -1)
            if j >= 0:
                x[i, j] = v
    return x.tocsr()


def classify_by_timediff(p, n, dTE):
    return np.where(np.logical_and(p[0] <= dTE, dTE <= p[1]),
                    1,
                    np.where(np.logical_and(n[0] <= dTE, dTE <= n[1]),
                             -1,
                             0))


def classify_normal(p, n, dTE, is_unknown, in_test_set):
    return np.where(is_unknown, -1, classify_by_timediff(p, n, dTE))


def classify_baseline(p, n, dTE, is_unknown, in_test_set):
    return np.where(is_unknown, -1,
                    np.where(in_test_set,
                             classify_by_timediff(p, n, dTE),
                             1))


def MakePredictions(w, xt, devData, wp_edits):
    predictions = []
    for i, fields in enumerate(devData):
        x = fields[0]
        y = fields[1]

        tweet = fields[-1]
        arg1 = tweet.arg1
        arg2 = tweet.arg2

        prediction = np.squeeze(lrxr.lr_predict(w, xt[i, :]))
        prediction = float(prediction)

        ep = "%s\t%s" % (arg1, arg2)
        wp_edits_ep = list(wp_edits.get(ep, []))

        editDate = None
        if len(wp_edits_ep) > 0:
            editDate = datetime.fromtimestamp(json.loads(wp_edits_ep[0])['timestamp'])
        predictions.append({'y': y, 'pred': prediction,
                            'arg1': arg1, 'arg2': arg2,
                            'tweetDate': tweet.datetime,
                            'editDate': editDate,
                            'tweet': tweet, 'wpEdits': wp_edits_ep})
    return predictions


AttrGroupData = namedtuple(
    'AttrGroupData',
    ['attributes', 'tweets_by_ep', 'wp_edits',
     'train_tweets', 'dev_tweets', 'test_tweets'])


def read_positive(base_path, attr_group,
                  train_range, dev_range,
                  train_tweets, dev_tweets,
                  ep_tweets, wp_edits,
                  labeled_ids):
    for attr in attr_group:
        print "reading %s" % attr
        idEps = set([])

        for line in open(os.path.join(base_path, attr)):
            (label, rel, arg1, arg2, tweetDate, editDate,
             timeSinceTweet, tweetStr, editStr) = line.strip().split('\t')

            timeSinceTweet = float(timeSinceTweet)

            edit = json.loads(editStr)
            title = edit['title']

            tweet = json.loads(tweetStr)

            del tweet['entity']
            del tweet['eType']
            del tweet['loc']
            del tweet['uid']
            del tweet['eventTags']
            if 'from_date' in tweet:
                del tweet['from_date']

            tweet = Tweet(
                arg1=arg1, arg2=arg2,
                title=title, from_negative=False,
                time_since_tweet=timeSinceTweet,
                datetime=ParseDate(tweet['created_at']),
                y=-1,
                **tweet)

            idEp = tweet.sid + '\t' + arg1 + '\t' + arg2
            if idEp in idEps:
                continue
            idEps.add(idEp)

            ep_tweets.setdefault(ep, []).append(tweet)

            # Keep track of the IDs in the labeled dataset so we can exclude
            # them from the unlabeled data.
            labeled_ids.add(tweet.sid)

            isInTrainRange = inside_range(train_range, tweet.datetime)
            isInDevRange   = inside_range(dev_range, tweet.datetime)
            if isInTrainRange:
                train_tweets.append(tweet)
            elif isInDevRange:
                dev_tweets.append(tweet)

            wp_edits.setdefault(ep, set()).add(editStr)

    print "sorting train"
    train_tweets.sort(tweet_datetime_cmp)
    print "sorting dev"
    dev_tweets.sort(tweet_datetime_cmp)


def read_data(attribute_groups, train_range, dev_range, test_range):
    attr_data = {}
    for attr_group in self.attribute_groups + [('negative',)]:
        attr_data[attr_group] = AttrGroupData(
            attributes=attr_group, tweets_by_ep={}, wp_edits={},
            train_tweets=[], dev_tweets=[], test_tweets=[])

    labeled_ids = set([])

    for attr_group in self.attribute_groups:
        data = attr_data[attr_group]

        gs.read_positive(
            base_dir="../training_data2/",
            attr_group=attr_group,
            train_range=train_range,
            dev_range=dev_range,
            test_range=test_range,
            data.train_tweets, dev_tweets,
            ep_tweets, wp_edits,
            labeledIds)

        print "done reading %s" % attr_group_name
        print "len(trainTweets)=%s" % len(train_tweets)
        print "len(devTweets)=%s" % len(dev_tweets)

    for attr_group in ['negative']:
        attr_group_name = 'negative'
        ep_tweets    = self.epTweets[attr_group_name]
        train_tweets = self.trainTweets[attr_group_name]
        dev_tweets   = self.devTweets[attr_group_name]

        print "reading negative"

        gs.read_preprocessed_negative(self.NEGATIVE_FILE,
            self.TRAIN_RANGE, self.DEV_RANGE,
            train_tweets, dev_tweets,
            ep_tweets, labeledIds)

        print "done reading %s" % attr_group_name
        print "len(trainTweets)=%s" % len(train_tweets)
        print "len(devTweets)=%s" % len(dev_tweets)

    table = []
    attrs = list(set(self.trainTweets.keys()) - set(['negative'])) + ['negative']
    for a in attrs:
        table.append((a, len(self.trainTweets[a]), len(self.devTweets[a])))
    print tabulate(table, headers=['attr', 'train', 'dev'])


def Run(self, output_dir):
    nsr = '1'
    self.ClassifyAll("normal")

    last_time = easytime.now()

    for attr_group in self.attribute_groups:
        attr_group_name = ','.join(attr_group)

        ep_tweets     = self.epTweets[attr_group_name]
        ep_tweets_neg = self.epTweets['negative']

        wp_edits     = self.wpEdits[attr_group_name]

        train_tweets = list(fx.merge_many(
            lambda s: s.datetime,
            self.trainTweets['negative'],
            self.trainTweets[attr_group_name]))
        dev_tweets   = list(fx.merge_many(
            lambda s: s.datetime,
            self.devTweets['negative'],
            self.devTweets[attr_group_name]))

        attr_all_file = open(os.path.join(output_dir, "%s.gs" % attr_group_name), 'w+')

        for fwd in self.FEATURE_WINDOW_DAYS:
            trainData, devData = self.ExtractFeatures(train_tweets, dev_tweets, fwd, ep_tweets, ep_tweets_neg)

            for mode, xr in itertools.product(self.MODES, self.XR_MODES):
                print "Reclassifying"
                for sample in devData:
                    tweet = sample[-1]
                    tweet.y = classify_tweet(self.POSITIVE_RANGE, self.NEGATIVE_RANGE,
                        mode, tweet.from_negative, True, tweet.time_since_tweet)
                    sample[1] = tweet.y

                xf = [v[0] for v in trainData if not v[-1].from_negative]
                dt = np.array([v[-1].time_since_tweet / easytime.dt(days=1)
                               for v in trainData
                               if not v[-1].from_negative])
                uf = [v[0] for v in trainData if v[-1].from_negative]

                vocabulary, reverse_vocabulary = \
                    make_vocabularies(fx.concat(xf, uf))
                print "n features: %s" % len(reverse_vocabulary)

                x = feature_matrix(xf, vocabulary, len(reverse_vocabulary))
                u = feature_matrix(uf, vocabulary, len(reverse_vocabulary))
                y = np.squeeze(classify_by_timediff(
                    self.POSITIVE_RANGE,
                    self.NEGATIVE_RANGE, -dt))
                xt = feature_matrix([v[0] for v in devData],
                                    vocabulary, len(reverse_vocabulary))

                ux = feature_matrix([v[0] for v in trainData],
                                    vocabulary, len(reverse_vocabulary))
                uy = classify_tweet(
                    self.POSITIVE_RANGE,
                    self.NEGATIVE_RANGE,
                    mode,
                    np.array([v[-1].from_negative for v in trainData]),
                    False,
                    np.array([v[-1].time_since_tweet for v in trainData]))
                xr1 = 10 * u.shape[1]
                l2 = 100.0
                t = 1.0

                # w_noem_0 = numpy.random.rand(x.shape[1])

                # for p_ex, iterations in fx.product(self.EXPECTATIONS, xrange(11)):
                #     w00, _, _ = lrxr.lrxr_train(
                #         w_noem_0,
                #         x, y, u, l2, t, p_ex, xr1, 1E-6)

                #     w0 = numpy.random.rand(x.shape[1] + 3)
                #     w0[:3] = [0.99, 0.0, 500]
                #     w0[3:] = w_noem_0
                #     w1 = numpy.random.rand(x.shape[1] + 3)
                #     w1[:3] = [0.01, 0.0, 10]
                #     w0[3:] = w_noem_0

                #     w1, w0 = lrxr.lrxrt_train(
                #         w1, w0, x, dt, u, l2, t, p_ex, xr1, 1E-6, iterations)
                #     pi1, mu1, s1 = tuple(w1[:3])

                w_noem_0 = numpy.random.rand(x.shape[1])
                w_em_0 = numpy.zeros(x.shape[1] + 6)
                w_em_0[6:] = w_noem_0
                w_em_0[:3] = [0.99, 0.0, 500]
                w_em_0[3:6] = [0.01, 0.0, 10]

                for p_ex, iterations in fx.product(self.EXPECTATIONS, xrange(11)):
                    w_noem_lrxr, _, _ = lrxr.lrxr_train(
                        w_noem_0,
                        x, y, u, l2, t, p_ex, xr1, 1E-6)
                    w_noem_lr, _, _ = lrxr.lr_train(
                        w_noem_0,
                        ux, uy,
                        l2, 1E-6)

                    w_em = lrxr.lrxrtv2_train(
                        w_em_0, x, dt, u, l2, t, p_ex, xr1, 1E-6, iterations)
                    pi1, mu1, s1 = tuple(w_em[3:6])

                    done_time = easytime.now()
                    time = done_time - last_time
                    last_time = done_time

                    for sample in devData:
                        tweet = sample[-1]
                        tweet.y = classify_tweet(self.POSITIVE_RANGE, self.NEGATIVE_RANGE,
                            mode, tweet.from_negative, True, tweet.time_since_tweet)
                        sample[1] = tweet.y

                    predictions_em = MakePredictions(w_em[6:], xt, devData, wp_edits)
                    predictions_em.sort(lambda a,b: cmp(b['pred'], a['pred']))
                    tprf_em = list(gen_tprf(predictions_em))
                    max_f1_em = max(t[-1] for t in tprf_em)

                    predictions_noem_lrxr = MakePredictions(w_noem_lrxr, xt, devData, wp_edits)
                    max_f1_noem_lrxr = max(t[-1] for t in gen_tprf(predictions_noem_lrxr, sort=True))

                    predictions_noem_lr = MakePredictions(w_noem_lr, xt, devData, wp_edits)
                    max_f1_noem_lr = max(t[-1] for t in gen_tprf(predictions_noem_lr, sort=True))

                    for sample in devData:
                        tweet = sample[-1]
                        mu1s = mu1 * easytime.dt(days=1)
                        tweet.y = classify_tweet(self.POSITIVE_RANGE, self.NEGATIVE_RANGE,
                            mode, tweet.from_negative, True, tweet.time_since_tweet - mu1s)
                        sample[1] = tweet.y
                    predictions_em1 = MakePredictions(w_em[6:], xt, devData, wp_edits)
                    max_f1_em1 = max(t[-1] for t in gen_tprf(predictions_em1, sort=True))

                    for sample in devData:
                        tweet = sample[-1]
                        mu1s = (mu1 - 6.5) * easytime.dt(days=1)
                        tweet.y = classify_tweet(self.POSITIVE_RANGE, self.NEGATIVE_RANGE,
                            mode, tweet.from_negative, True, tweet.time_since_tweet - mu1s)
                        sample[1] = sample[-1].y
                    predictions_em2 = MakePredictions(w_em[6:], xt, devData, wp_edits)
                    max_f1_em2 = max(t[-1] for t in gen_tprf(predictions_em2, sort=True))

                    subdir = "%s_%s_%s_%s_%s_%s" % (attr_group_name, mode, xr + str(iterations), fwd, nsr, p_ex)
                    gs.mkdir_p(os.path.join(output_dir, subdir))

                    with open(os.path.join(output_dir, subdir, 'PRout'), 'w+') as f:
                        for T, P, R, F in tprf_em:
                            f.write("%s\t%s\t%s\t%s\n" % (T, P, R, F))

                    print "%s/%s/%s/%s: F1=%.2f %.2f / %.2f %.2f %.2f" % \
                        (attr_group_name, mode, xr + str(iterations), p_ex,
                            max_f1_noem_lr, max_f1_noem_lrxr,
                            max_f1_em, max_f1_em1, max_f1_em2)

                    with open(os.path.join(output_dir, subdir, 'paramOut'), 'w+') as f:
                        f.write("mode                 = %s\n" % mode)
                        f.write("xr_mode              = %s\n" % xr + str(iterations))
                        f.write("feature_window_days  = %s\n" % fwd)
                        f.write("negative_sample_rate = %s\n" % nsr)
                        f.write("p_ex                 = %s\n" % p_ex)
                        f.write("max_f1_noem_lr       = %s\n" % max_f1_em)
                        f.write("max_f1_noem_lrxr     = %s\n" % max_f1_em)
                        f.write("max_f1_em            = %s\n" % max_f1_em)
                        f.write("max_f1_em1           = %s\n" % max_f1_em1)
                        f.write("max_f1_em2           = %s\n" % max_f1_em2)
                        f.write("vocabulary           = %s\n" % len(reverse_vocabulary))

                    with open(os.path.join(output_dir, subdir, 'predOut'), 'w+') as f:
                        for p in predictions_em:
                            p['tweet'] = p['tweet']._asdict()
                            del p['editDate']
                            # del p['tweetDate']
                            # del p['tweet']['datetime']
                            p['tweet']['y'] = float(p['tweet']['y'])
                            p['y'] = float(p['y'])
                            #print p
                            f.write(json.dumps(p) + "\n")

                    with open(os.path.join(output_dir, subdir, 'weights'), 'w+') as f:
                        pi, mu, s = w_em[3:6]
                        w = w_em[6:]

                        tsv_print(f, pi, mu, s)
                        for i in np.argsort(-w_em[6:]):
                            tsv_print(f, reverse_vocabulary[i], w_em[6 + i], w_em[6 + i] - w_noem_lrxr[i])

                    # with open(os.path.join(output_dir, subdir, 'weights_n'), 'w+') as f:
                    #     pi, mu, s = w0[:3]
                    #     w = w0[3:]

                    #     tsv_print(f, pi, mu, s)
                    #     for i in np.argsort(-w[3:]):
                    #         tsv_print(f, reverse_vocabulary[i], w[3 + i])

                    tsv_print(attr_all_file, attr_group_name, mode,
                              xr + str(iterations), fwd, nsr,
                              "%.1f" % (-np.log(p_ex) / np.log(10)),
                              "%.3f" % max_f1_noem_lr,
                              "%.3f" % max_f1_noem_lrxr,
                              "%.3f" % max_f1_em,
                              "%.3f" % max_f1_em1,
                              "%.3f" % max_f1_em2)
                    attr_all_file.flush()
        attr_all_file.close()


MIN_FEATURE_COUNT = 3
MIN_TWEET_COUNT = 3

MODES = ["normal"]
XR_MODES = ["lrxrt"]
FEATURE_WINDOW_DAYS = [1, 2, 4, 8, 16, 24, 32]
EXPECTATIONS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

TRAIN_RANGE = (datetime(year=2008, month=9, day=1),
               datetime(year=2011, month=6, day=1))
DEV_RANGE = (datetime(year=2011, month=6, day=5),
             datetime(year=2011, month=8, day=1))

POSITIVE_RANGE = [-10, 3]
NEGATIVE_RANGE = [-50, 3]


def run(attributes, output_dir, negative_file):
    ds = GridSearch(attributes)
    ds.Run(output_dir)


def main(attributes, output_dir, parallel, debug):
    if debug:
        attributes = [['death_place']]
        negative_file = "../training_data2/negative_small.gz"
        parallel = False
    else:
        negative_file = "../training_data2/negative_1.gz"

    if parallel:
        pool = mp.Pool(16)
        pool.map(run, [[x, output_dir, negative_file] for x in attributes])
    else:
        run(attributes, output_dir, negative_file)


if __name__ == "__main__":
    PROFILE = False
    PARALLEL = True
    DEBUG = False

    profiler = None
    if PROFILE:
        profiler = cProfile.Profile()
        profiler.enable()

    attributes = [tuple(s.split(',')) for s in sys.argv[1].split(';')]
    main(attributes, sys.argv[2], PARALLEL, DEBUG)

    if PROFILE:
        profiler.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
