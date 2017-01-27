print "wtf"

import sys, errno, os, os.path
import LR
import ujson as json
import gzip
from datetime import *
from FeatureExtractor import *
import cProfile
import itertools

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'new'))
import easytime, fx

sys.path.append('../../weakly_supervised_events/python')

#from HDEM import *
from Classifier import *
from Vocab import *

def ParseDate(string):
    return datetime.strptime(string, '%a %b %d %H:%M:%S +0000 %Y')

def inside_range(range, x):
    return range[0] <= x and x <= range[1]

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

from recordtype import recordtype
from tabulate import tabulate
Tweet = recordtype('Tweet', ['arg1', 'arg2', 'sid', 'words', 'pos',
                             'neTags', 'date', 'y', 'datetime',
                             'from_negative', 'time_since_tweet',
                             'title', 'created_at'])

class GridSearch:
    MIN_FEATURE_COUNT = 3
    MIN_TWEETS        = 3

    MODES=["normal", "baseline"]
    XR_MODES=["lr", "lrxr"]
    FEATURE_WINDOW_DAYS=[1, 2, 4, 8, 16, 32]
    EXPECTATIONS=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    NEGATIVE_FILE="../training_data2/negative_1.gz"
    TRAIN_RANGE=(datetime(year=2008,month=9,day=1), datetime(year=2011,month=6,day=1))
    DEV_RANGE=(datetime(year=2011,month=6,day=5), datetime(year=2011,month=8,day=1))

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

    def __init__(self, attribute_groups):
        self.attribute_groups = attribute_groups

        self.epTweets    = {}#{}
        self.trainTweets = {}#[]
        self.devTweets   = {}#[]
        self.wpEdits     = {}#{}

        labeledIds = set([])

        datetime_cmp = lambda a,b: cmp(a.datetime, b.datetime)

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

            for attr in attr_group:
                print "reading %s" % attr
                idEps = set([])
                for line in open("../training_data2/" + attr):
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

                    ep  = "%s\t%s" % (arg1,arg2)
                    ep_tweets.setdefault(ep, []).append(tweet)

                    ################################################################################################
                    #Keep track of the IDs in the labeled dataset so we can exclude them from the unlabeled data...
                    ################################################################################################
                    labeledIds.add(tweet.sid)

                    #self.allTweets.append(tweet)
                    isInTrainRange = inside_range(self.TRAIN_RANGE, tweet.datetime)
                    isInDevRange   = inside_range(self.DEV_RANGE, tweet.datetime)
                    if isInTrainRange:
                        train_tweets.append(tweet)
                    elif isInDevRange:
                        dev_tweets.append(tweet)

                    wp_edits.setdefault(ep, set()).add(editStr)

            print "sorting train"
            train_tweets.sort(datetime_cmp)
            print "sorting dev"
            dev_tweets.sort(datetime_cmp)

            print "done reading %s" % attr_group_name
            print "len(trainTweets)=%s" % len(train_tweets)
            print "len(devTweets)=%s" % len(dev_tweets)

        nNeg = 0
        idEps = set([])

        if True:
            attr_group_name = 'negative'
            ep_tweets    = self.epTweets[attr_group_name]
            train_tweets = self.trainTweets[attr_group_name]
            dev_tweets   = self.devTweets[attr_group_name]

            print "reading negative"

            for line in gzip.open(self.NEGATIVE_FILE):
                nNeg += 1
                if nNeg % 100000 == 0:
                    print "number of negative read: %s" % nNeg

                tweet = json.loads(line)

                tweet = Tweet(
                    # arg1=arg1, arg2=arg2,
                    title=arg1, from_negative=True,
                    time_since_tweet=0.0,
                    datetime=datetime.fromtimestamp(tweet['created_at']),
                    y=-1,
                    **tweet)

                isInTrainRange = inside_range(self.TRAIN_RANGE, tweet.datetime)
                isInDevRange   = inside_range(self.DEV_RANGE, tweet.datetime)

                if not tweet.sid in labeledIds:
                    ep  = "%s\t%s" % (tweet.arg1,tweet.arg2)
                    ep_tweets.setdefault(ep, []).append(tweet)

                    if isInTrainRange:
                        train_tweets.append(tweet)
                    elif isInDevRange:
                        dev_tweets.append(tweet)

            print "sorting train"
            train_tweets.sort(datetime_cmp)
            print "sorting dev"
            dev_tweets.sort(datetime_cmp)

            print "done reading %s" % attr_group_name
            print "len(trainTweets)=%s" % len(train_tweets)
            print "len(devTweets)=%s" % len(dev_tweets)

        table = []
        attrs = list(set(self.trainTweets.keys()) - set(['negative'])) + ['negative']
        for a in attrs:
            table.append((a, len(self.trainTweets[a]), len(self.devTweets[a])))
        print tabulate(table, headers=['attr', 'train', 'dev'])

    def ListPRF(self, predictions, N):
        """ Computes a single precision and recall of a list of predictions """
        tp = 0.0
        fp = 0.0
        fn = 0.0
        result = []

        for p in predictions:
            T = p['pred']
            if p['y'] == 1:
                tp += 1
            elif p['y'] == -1:
                fp += 1

            fn = N - tp

            P = 0
            if tp + fp > 0:
                P = tp / (tp + fp)
            R = 0
            if tp + fn > 0:
                R = tp / (tp + fn)
            F = 0
            if P + R > 0:
                F = 2 * P * R / (P + R)

            result.append((T, P, R, F))
        return result

    def MaxF1(self, predictions, N):
        """ Computes maximum F1 """
        tp = 0.0
        fp = 0.0
        fn = 0.0
        maxF = 0.0

        for p in predictions:
            if p['y'] == 1:
                tp += 1
            elif p['y'] == -1:
                fp += 1

            fn = N - tp

            P = 0
            if tp + fp > 0:
                P = tp / (tp + fp)
            R = 0
            if tp + fn > 0:
                R = tp / (tp + fn)
            F = 0
            if P + R > 0:
                F = 2 * P * R / (P + R)

            if maxF < F:
                maxF = F

        return maxF

    def MakePredictions(self, model, devData, wp_edits):
        predictions = []
        for fields in devData:
            x = fields[0]
            y = fields[1]
            prediction = model.Predict(x)
            #tweetDate = fields[2]
            tweet = fields[5]
            #tweet.datetime = ParseDate(tweet.created_at)
            #tweetDate = datetime.fromtimestamp(tweet.created_at)
            arg1 = fields[3]
            arg2 = fields[4]
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
                    self.ClassifyAll(mode)

                    for v in itertools.chain(iter(trainData), iter(devData)):
                        # [fe.Features(), target.y, tweetDate, title, arg2, target]
                        target = v[-1]
                        v[1] = target.y

                    # P, D, U, N = self.PDU(trainData)
                    # print "P = %.5f" % P
                    # print "D = %.5f" % D
                    # print "U = %.5f" % U
                    # print "N = %.5f" % N

                    model = LR_XRclassifierV2() if xr == "lrxr" else LRclassifierV2()
                    model.Prepare([x for x in trainData if x[1] != 0])
                    print "n features: %s" % model.vocab.GetVocabSize()

                    for p_ex in self.EXPECTATIONS if xr == "lrxr" else [0]:
                        if xr == "lrxr":
                            model.Train(p_ex=p_ex, l2=100.0)
                        else:
                            model.Train(l2=100.0)

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
                        PR = self.ListPRF(predictions, N)
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
        prevDates = {}

        for target in tweets:
            arg1 = target.arg1
            arg2 = target.arg2
            ep = "%s\t%s" % (arg1, arg2)

            prevDates[ep] = target.datetime
            other_tweets = ep_tweets.get(ep, []) + ep_tweets_neg.get(ep, [])

            if len(other_tweets) >= self.MIN_TWEETS:
                max_time = target.datetime
                min_time = target.datetime - timedelta(days=feature_window_days)
                predicate = lambda x: x.datetime <= max_time and \
                                      x.datetime > min_time
                window = [x for x in other_tweets if predicate(x)]

                fe = FeatureExtractorBinary(window, target, arg1, arg2)
                fe.AddBiasFeature()
                fe.ComputeBinaryFeatures()
                fe.ComputeTimexFeatures()
                # fe.ComputeEntityFeatures()
                # fe.ComputeVerbFeatures()
                tweetDate = target.datetime.strftime("%Y-%m-%d")
                result.append([fe.Features(), target.y, tweetDate, arg1, arg2, target])

        return result

    def PDU(self, data):
        total = 0
        positive = 0
        discarded = 0
        negative = 0
        unknown = 0

        for tweet in data:
            if tweet[1] == 1:
                positive += 1
            elif tweet[1] == -1:
                unknown += 1
            else:
                discarded += 1

            if tweet[-1].from_negative:
                negative += 1

            total += 1

        P = positive / float(total)
        D = discarded / float(total)
        U = unknown / float(total)
        N = negative / float(total)

        return (P, D, U, N)

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

    def ExtractFeatures(self, trainTweets, devTweets, feature_window_days,
                        ep_tweets, ep_tweets_neg):
        print "Extracting Features"
        trainData = self.GenData(trainTweets, feature_window_days, ep_tweets, ep_tweets_neg)
        print "len(trainData)=%s" % len(trainData)
        fs = FeatureSelectionEntity(trainTweets, trainData)
        trainData = fs.FilterFeaturesByCount(self.MIN_FEATURE_COUNT)

        devData = self.GenData(devTweets, feature_window_days, ep_tweets, ep_tweets_neg)
        print "Done Extracting Features"
        print "len(self.train)=%s" % len(trainData)
        print "len(self.dev)=%s" % len(devData)

        return (trainData, devData)

PROFILE=False
PARALLEL=True

import multiprocessing as mp

def run(attr):
    ds = GridSearch([attr])
    ds.Run(sys.argv[2])

if __name__ == "__main__":

    pr = None
    if PROFILE:
        pr = cProfile.Profile()
        pr.enable()

    attrs = [s.split(',') for s in sys.argv[1].split(';')]

    if PARALLEL:
        pool = mp.Pool(16)
        pool.map(run, attrs)
    else:
        ds = GridSearch(attrs)
        ds.Run(sys.argv[2])

    if PROFILE:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
