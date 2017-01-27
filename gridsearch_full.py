import sys, errno, os, os.path
import LR
import ujson as json
import gzip
from datetime import *
from FeatureExtractor import *
import cProfile
import itertools
import re
import gc
import bisect

import fx
import easytime
import lrxr

sys.path.append('../../weakly_supervised_events/python')

#from HDEM import *
from Classifier import *
from Vocab import *
from gs_utils import ParseDate, inside_range, mkdir_p, \
    read_preprocessed_negative, read_positive, tsv_print, ListPRF, MaxF1, \
    tweet_datetime_cmp, PDNU

from collections import namedtuple
from recordtype import recordtype
from tabulate import tabulate


class LRXRTA(object):
    def __init__(self):
        self.model = None

    def Prepare(self, data, vocabulary, index):
        print "-> LRXRTA.Prepare"

        self.vocabulary = vocabulary
        self.index = index

        Y = np.zeros(len(data))
        X_matrix = lil_matrix((len(data), len(self.vocabulary)))
        for i, sample in enumerate(data):
            Y[i] = sample[1]
            for j in sample[0]:
                X_matrix[i, j] = 1
        X_matrix = X_matrix.tocsr()

        pos_indices = (Y ==  1).nonzero()[0]    # positive (need to do this for sparse matrices)
        unl_indices = (Y == -1).nonzero()[0]    # unlabeled
        self.X = X_matrix[pos_indices,:]
        self.Y = Y[Y==1]
        self.U = X_matrix[unl_indices, :]

        print "<- LRXRTA.Prepare"

    def Train(self, p_ex=0.5, temp=1.0, l2=1.0, xr=10.0):
        print "-> LRXRTA.Train"

        mu0, s20 = 0, 400**2
        mu1, s21 = 0, 3**2

        w0 = np.random.randn(self.X.shape[1])

        lrxr.lrdem_train(
            np.concatenate(([mu0, s20, mu1, s21], w0)), x, dt, u,
            l2=1.0, t=1.0, p_ex=0.1, xr=10.0*u.shape[0], tol=1e-10)

        lr = LR_XR(self.X, self.Y, self.U, p_ex=p_ex, temp=temp, xr=xr, l2=l2)
        lr.Train()

        print lr.nll
        print lr.status

        self.model = lr

        print "<- LRXRTA.Train"

    def Predict(self, data):
        return self.model.Predict(densify(data, len(self.vocabulary)))

    def PrintWeights(self, outFile):
        fOut = open(outFile, 'w')
        for i in np.argsort(-self.model.wStar):
            fOut.write("%s\t%s\n" % (self.vocabulary[i], self.model.wStar[i]))


class Vocabulary(object):
    __slots__ = ('words', 'index')

    def __init__(self, words=None, index=None):
        self.words = words if words is not None else []
        if index is not None:
            self.index = index
        else:
            self.index = dict(map(reversed, enumerate(self.words)))

    def update(self, w):
        if w not in self.index:
            self.index[w] = len(self.words)
            self.words.append(w)
            return len(self.words) - 1
        return self.index[w]

    def get(self, w):
        return self.index.get(w, -1)

    def resolve(self, w, immutable=False):
        self.get(voc, w) if immutable else self.update(voc, w)

    def compress(self, predicate):
        new_words = []
        new_index = {}
        remapping = [0] * len(self.words)

        for i, w in enumerate(self.words):
            if predicate(i, w):
                new_index[w] = len(new_words)
                remapping[i] = len(new_words)
                new_words.append(w)
            else:
                remapping[i] = -1

        self.index = new_index
        self.words = new_words

        return remapping

    def __len__(self):
        return len(self.words)


class GridSearch:
    MIN_FEATURE_COUNT = 5
    MIN_TWEETS = 5

    MODES = ["normal", "baseline"]
    XR_MODES = ["lr"]
    FEATURE_WINDOW_DAYS = [1]
    EXPECTATIONS = [1, 2, 4, 8, 0.5, 0.25, 0.125]

    NEGATIVE_FILE = "../training_data2/negative_1.gz"
    TRAIN_RANGE = (datetime(year=2008, month=9, day=1),
                   datetime(year=2011, month=6, day=1))
    DEV_RANGE = (datetime(year=2011, month=6, day=5),
                 datetime(year=2012, month=1, day=1))

    POSITIVE_RANGE = [-10, 3]
    NEGATIVE_RANGE = [-50, 3]

    def classify_by_timediff(self, time_since_tweet):
        oneDay = 24 * 60 * 60.0
        T = 0
        E = time_since_tweet / oneDay

        if inside_range(self.POSITIVE_RANGE, T - E):
            return 1
        elif not inside_range(self.NEGATIVE_RANGE, T - E):
            return -1
        else:
            return 0

    def classify_tweet(self, mode, tweet, negative, in_test_set, time_since_tweet):
        if negative:
            tweet.y = -1
            return

        if mode == "normal":
            tweet.y = self.classify_by_timediff(time_since_tweet)
        elif mode == "baseline":
            if in_test_set:
                tweet.y = self.classify_by_timediff(time_since_tweet)
            else:
                tweet.y = 1
        else:
            assert False

    def __init__(self, attribute_groups, infoboxes):
        self.attribute_groups = attribute_groups

        self.epTweets    = {}#{}
        self.trainTweets = {}#[]
        self.devTweets   = {}#[]
        self.wpEdits     = {}#{}

        labeledIds = set([])

        for attr_group in self.attribute_groups + [['negative']]:
            attr_group_name = ','.join(attr_group)
            self.epTweets[attr_group_name]    = {}
            self.trainTweets[attr_group_name] = []
            self.devTweets[attr_group_name]   = []
            self.wpEdits[attr_group_name]     = {}

        for attr_group in self.attribute_groups:
            attr_group_name = ','.join(attr_group)
            ep_tweets    = self.epTweets[attr_group_name]
            train_tweets = self.trainTweets[attr_group_name]
            dev_tweets   = self.devTweets[attr_group_name]
            wp_edits     = self.wpEdits[attr_group_name]

            read_positive("../training_data2/", attr_group,
                self.TRAIN_RANGE, self.DEV_RANGE,
                train_tweets, dev_tweets,
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

            read_preprocessed_negative(self.NEGATIVE_FILE,
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

    def MakePredictions(self, model, devData, wp_edits):
        predictions = []
        for fields in devData:
            x = fields[0]
            y = fields[1]
            prediction = model.Predict(x)
            tweet = fields[-1]
            arg1 = tweet.arg1
            arg2 = tweet.arg2
            ep = "%s\t%s" % (arg1,arg2)
            #print ep
            wp_edits_ep = list(wp_edits.get(ep, []))
            editDate = None
            if len(wp_edits_ep) > 0:
                editDate = datetime.fromtimestamp(json.loads(wp_edits_ep[0])['timestamp'])
            predictions.append({'y':y, 'pred':prediction,
                                'arg1':arg1, 'arg2':arg2,
                                'tweetDate':tweet.datetime,
                                'editDate':editDate,
                                'tweet':tweet , 'wpEdits':wp_edits_ep})

        return predictions

    def Run(self, output_dir):
        nsr = 'all'
        self.ClassifyAll("normal")

        # print "Preprocessing/SortingEPMaps started."
        # for tweets in ep_tweets.itervalues():
        #     tweets.sort(tweet_datetime_cmp)
        # print "Preprocessing/SortingEPMaps is done."

        print "Preprocessing/ComputeEpFrequencies started."
        ep_counts = {}
        for attr_group in self.attribute_groups + [['negative']]:
            attr_group_name = ','.join(attr_group)
            train_tweets = self.trainTweets[attr_group_name]
            dev_tweets = self.devTweets[attr_group_name]
            for tweet in fx.concat(train_tweets, dev_tweets):
                ep = (tweet.arg1, tweet.arg2)
                ep_counts[ep] = ep_counts.setdefault(ep, 0) + 1
        print "Preprocessing/ComputeEpFrequencies is done."

        print "Preprocessing/DropEps started."
        discard_eps = set()
        for ep, cnt in ep_counts.iteritems():
            if cnt < self.MIN_TWEETS:
                discard_eps.add(ep)
        ep_counts = None

        for attr_group in self.attribute_groups + [['negative']]:
            attr_group_name = ','.join(attr_group)
            self.trainTweets[attr_group_name] = \
                [x for x in self.trainTweets[attr_group_name]
                 if (x.arg1, x.arg2) not in discard_eps]
            self.devTweets[attr_group_name] = \
                [x for x in self.devTweets[attr_group_name]
                 if (x.arg1, x.arg2) not in discard_eps]

            ep_tweets = self.epTweets[attr_group_name]
            for ep in discard_eps:
                ep_str = '%s\t%s' % ep
                if ep_str in ep_tweets:
                    del ep_tweets[ep_str]
        discard_eps = None
        print "Preprocessing/DropEps is done."

        table = []
        attrs = list(set(self.trainTweets.keys()) - set(['negative'])) + ['negative']
        for a in attrs:
            table.append((a, len(self.trainTweets[a]), len(self.devTweets[a])))
        print tabulate(table, headers=['Attr', 'Train Samples', 'Test Samples'], tablefmt='latex')

        print "Preprocessing/ExtractingFeatures started."
        NO_UNLABELED_FEATURES = False
        vocabulary = Vocabulary(['__BIAS__'])
        index = {}

        for attr_group in self.attribute_groups + [['negative']]:
            attr_group_name = ','.join(attr_group)
            train_tweets = self.trainTweets[attr_group_name]
            dev_tweets = self.devTweets[attr_group_name]
            for tweet in fx.concat(train_tweets, dev_tweets):
                args = (tweet.arg1, tweet.arg2)
                words = tweet.words.split(' ')
                pos = tweet.pos
                neTags = tweet.neTags

                features = fx.concat(
                    ['__BIAS__'], generate_binary_features(
                        args, words, pos, neTags))
                # features = generate_binary_features(tweet)
                if attr_group_name == 'negative' and NO_UNLABELED_FEATURES:
                    tweet.features = sorted(set(vocabulary.get(f)
                                                for f in features))
                else:
                    tweet.features = sorted(set(vocabulary.update(f)
                                                for f in features))
        print "n features: %s" % len(vocabulary)
        print "Preprocessing/ExtractingFeatures is done."

        print "Preprocessing/ComputeFeatureFrequencies started."
        feature_eps = [None] * len(vocabulary)
        for i, _ in enumerate(feature_eps):
            feature_eps[i] = set()

        for attr_group in self.attribute_groups + [['negative']]:
            attr_group_name = ','.join(attr_group)
            train_tweets = self.trainTweets[attr_group_name]
            for tweet in train_tweets:
                for feature in tweet.features:
                    ep = (tweet.arg1, tweet.arg2)
                    feature_eps[feature].add(ep)
        for i, _ in enumerate(feature_eps):
            feature_eps[i] = len(feature_eps[i])
        print "Preprocessing/ComputeFeatureFrequencies is done."

        print "Preprocessing/RemapFeatures started."
        remapping = vocabulary.compress(
            lambda i, w: feature_eps[i] >= self.MIN_FEATURE_COUNT)

        for attr_group in self.attribute_groups + [['negative']]:
            attr_group_name = ','.join(attr_group)
            train_tweets = self.trainTweets[attr_group_name]
            dev_tweets = self.devTweets[attr_group_name]
            for tweet in fx.concat(train_tweets, dev_tweets):
                tweet.features = [remapping[i] for i in tweet.features
                                  if remapping[i] != -1]

        print "n features: %s" % len(vocabulary)
        remapping = None
        feature_eps = None
        print "Preprocessing/RemapFeatures is done."

        print "Preprocessing/UnlabeledWindows started."
        unlabeled_training_set = self.GenData(
            tweets=self.trainTweets['negative'],
            feature_window_days=1,
            ep_tweets=self.epTweets['negative'],
            ep_tweets_neg={})
        unlabeled_dev_set = self.GenData(
            tweets=self.devTweets['negative'],
            feature_window_days=1,
            ep_tweets=self.epTweets['negative'],
            ep_tweets_neg={})
        print "len(unlabeled_training_set)=%s" % len(unlabeled_training_set)
        print "len(unlabeled_dev_set)=%s" % len(unlabeled_dev_set)
        print "Preprocessing/UnlabeledWindows is done."

        # print "Preprocessing/ConsistencyChecking started."
        # for attr_group in self.attribute_groups + [['negative']]:
        #     attr_group_name = ','.join(attr_group)
        #     train_tweets = self.trainTweets[attr_group_name]
        #     dev_tweets = self.devTweets[attr_group_name]
        #     ep_tweets = self.epTweets[attr_group_name]
        #     for tweet in fx.concat(train_tweets, dev_tweets):
        #         assert tweet.features[0] == 0
        #         assert tweet.features is not None
        #     for ep, tweets in ep_tweets.iteritems():
        #         for tweet in tweets:
        #             assert tweet.features[0] == 0
        #             assert tweet.features is not None
        # print "Preprocessing/ConsistencyChecking is done."

        last_time = easytime.now()

        for attr_group in self.attribute_groups:
            attr_group_name = ','.join(attr_group)

            ep_tweets     = self.epTweets[attr_group_name]
            ep_tweets_neg = self.epTweets['negative']

            wp_edits     = self.wpEdits[attr_group_name]

            train_tweets = self.trainTweets[attr_group_name]
            dev_tweets = self.devTweets[attr_group_name]
            # train_tweets = fx.merge_many(
            #     self.trainTweets['negative'],
            #     self.trainTweets[attr_group_name],
            #     key=lambda s: s.datetime)
            # dev_tweets   = fx.merge_many(
            #     self.devTweets['negative'],
            #     self.devTweets[attr_group_name],
            #     key=lambda s: s.datetime)

            attr_all_file = open(os.path.join(output_dir, "%s.gs" % attr_group_name), 'w+')

            for fwd in self.FEATURE_WINDOW_DAYS:
                trainData, devData, model, predictions = None, None, None, None
                gc.collect()

                extract_start = easytime.now()
                # trainData, devData = self.ExtractFeatures(
                #     train_tweets, dev_tweets, fwd, ep_tweets, ep_tweets_neg)
                print "Extracting Features"
                trainData = self.GenData(
                    tweets=train_tweets,
                    feature_window_days=1,
                    ep_tweets=ep_tweets,
                    ep_tweets_neg={}) + unlabeled_training_set
                devData = self.GenData(
                    tweets=dev_tweets,
                    feature_window_days=1,
                    ep_tweets=ep_tweets,
                    ep_tweets_neg={}) + unlabeled_dev_set
                print "Done Extracting Features"
                print "Extraction took %s seconds." % (easytime.now() - extract_start)

                for mode, xr in itertools.product(self.MODES, self.XR_MODES):
                    gc.collect()
                    print "Reclassifying"
                    self.ClassifyAll(mode)

                    for v in itertools.chain(iter(trainData), iter(devData)):
                        # [fe.Features(), target.y, tweetDate, title, arg2, target]
                        target = v[-1]
                        v[1] = target.y

                    P, D, N, U = PDNU(trainData)
                    print "P = %.5f" % P
                    print "D = %.5f" % D
                    print "U = %.5f" % U
                    print "N = %.5f" % N

                    P, D, N, U = PDNU(devData)
                    print "P(dev) = %.5f" % P
                    print "D(dev) = %.5f" % D
                    print "U(dev) = %.5f" % U
                    print "N(dev) = %.5f" % N

                    model = LR_XRclassifierV2() if xr == "lrxr" else LRclassifierV2()
                    model.Prepare([x for x in trainData if x[1] != 0],
                                  vocabulary.words, vocabulary.index)

                    for p_ex in self.EXPECTATIONS if xr == "lrxr" else [0]:
                        p_ex = (P / 500) * p_ex

                        if xr == "lrxr":
                            model.Train(p_ex=p_ex, l2=10.0)
                        else:
                            model.Train(l2=10.0)

                        print("nll=", model.model.nll)
                        print("status=", model.model.status)

                        predictions = self.MakePredictions(model, devData, wp_edits)

                        N = sum(1 for x in predictions if x['y'] == 1)
                        predictions.sort(lambda a,b: cmp(b['pred'], a['pred']))
                        #F = self.MaxF1(predictions, N)

                        subdir = "%s_%s_%s_%s_%s_%s" % (attr_group_name, mode, xr, fwd, nsr, p_ex)
                        mkdir_p(os.path.join(output_dir, subdir))
                        paramOut   = open(os.path.join(output_dir, subdir, 'paramOut'), 'w+')
                        PRout      = open(os.path.join(output_dir, subdir, 'PRout'), 'w+')
                        maxFout    = open(os.path.join(output_dir, subdir, 'maxFout'), 'w+')
                        predOut    = open(os.path.join(output_dir, subdir, 'predOut'), 'w+')

                        maxF = 0
                        PR = ListPRF(predictions, N)
                        for i in range(len(predictions)):
                            (T, P, R, F) = PR[i]
                            PRout.write("%s\t%s\t%s\t%s\n" % (T, P,R,F))
                            if F > maxF:
                                maxF = F
                        PRout.close()

                        done_time = easytime.now()
                        time = done_time - last_time
                        last_time = done_time

                        print "%s/%s/%s/%s: F1=%s in %s s" % (attr_group_name, mode, xr, p_ex, maxF, time)

                        paramOut.write("mode                 = %s\n" % mode)
                        paramOut.write("xr_mode              = %s\n" % xr)
                        paramOut.write("feature_window_days  = %s\n" % fwd)
                        paramOut.write("negative_sample_rate = %s\n" % nsr)
                        paramOut.write("p_ex                 = %s\n" % p_ex)
                        paramOut.write("F                    = %s\n" % maxF)
                        paramOut.write("time                 = %s\n" % time)
                        paramOut.close()

                        maxFout.write(str(maxF) + "\n")
                        maxFout.close()

                        for p in predictions:
                            p['tweet'] = p['tweet']._asdict()
                            del p['editDate']

                            predOut.write(json.dumps(p) + "\n")
                        predOut.close()

                        model.PrintWeights(os.path.join(output_dir, subdir, 'weights'))

                        attr_all_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (attr_group_name, mode, xr, fwd, nsr, p_ex, maxF))
                        attr_all_file.flush()


            attr_all_file.close()

    #PRECONDITION: tweets must be sorted by time before calling GenData
    def GenData(self, tweets, feature_window_days, ep_tweets, ep_tweets_neg):
        result = []

        for target in tweets:
            arg1 = target.arg1
            arg2 = target.arg2
            ep = "%s\t%s" % (arg1, arg2)

            labeled = ep_tweets.get(ep, [])
            unlabeled = ep_tweets_neg.get(ep, [])

            #if len(labeled) + len(unlabeled) >= self.MIN_TWEETS:
            dt = easytime.dt(days=feature_window_days)
            predicate = lambda x: x.timestamp <= target.timestamp and \
                                  x.timestamp > (target.timestamp - dt)
            window = [x.features for x in fx.concat(labeled, unlabeled)
                      if predicate(x)]
            window_features = sorted(set(fx.concat(*window)))
            result.append([window_features, target.y, target])

        return result

    def ClassifyAll(self, mode):
        all_tweets = fx.concat(*(self.trainTweets.values() + self.devTweets.values()))
        neg_count = 0
        for tweet in all_tweets:
            isInTrainRange = inside_range(self.TRAIN_RANGE, tweet.datetime)
            isInDevRange   = inside_range(self.DEV_RANGE, tweet.datetime)
            self.classify_tweet(mode, tweet, tweet.from_negative, isInDevRange, tweet.time_since_tweet)
            if tweet.from_negative:
                tweet.y = -1
                neg_count += 1
        print "neg_count=%s" % neg_count

        # total = 0
        # positive = 0
        # discarded = 0
        # unknown = 0
        # for tweet in fx.concat(*(self.trainTweets.values() + self.devTweets.values())):
        #     if tweet.y == 1:
        #         positive += 1
        #     elif tweet.y == -1:
        #         unknown += 1
        #     else:
        #         discarded += 1
        #     total += 1

        # P = positive / float(total)
        # D = discarded / float(total)
        # U = unknown / float(total)

        # print "%s, %s, %s" % (P, D, U)

    # def ExtractFeatures(self, trainTweets, devTweets, feature_window_days,
    #                     ep_tweets, ep_tweets_neg):
    #     print "Extracting Features"
    #     trainData = self.GenData(trainTweets, feature_window_days,
    #                              ep_tweets, ep_tweets_neg)
    #     devData = self.GenData(devTweets, feature_window_days,
    #                            ep_tweets, ep_tweets_neg)
    #     print "Done Extracting Features"
    #     print "len(self.train)=%s" % len(trainData)
    #     print "len(self.dev)=%s" % len(devData)

    #     return (trainData, devData)

PROFILE=False
PARALLEL=False

import multiprocessing as mp

def run(params):
    attr, infoboxes = params
    ds = GridSearch([attr], infoboxes)
    ds.Run(sys.argv[2])

if __name__ == "__main__":

    pr = None
    if PROFILE:
        pr = cProfile.Profile()
        pr.enable()

    attrs = [s.split(',') for s in sys.argv[1].split(';')]
    infoboxes = [
        'infoboxsenator',
        'infoboxcongressman',
        #'infoboxgovernorelect',
        'infoboxpolitician',
        'infoboxltgovernor',
        'infoboxstatesenator',
        'infoboxofficeholder',
        'infoboxstaterepresentative',
        'infoboxcongressionalcandidate',
        #'infoboxcongressmanelect',
        'infoboxmayor',
        'infoboxspeaker',
        'infoboxgovernor',
        'infoboxcongresswoman',
        'infoboxuniversityundergraduate'
    ]

    if PARALLEL:
        pool = mp.Pool(16)
        pool.map(run, [(a, infoboxes) for a in attrs])
    else:
        ds = GridSearch(attrs, infoboxes)
        ds.Run(sys.argv[2])

    if PROFILE:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
