import sys, errno, os, os.path
import LR
import ujson as json
import gzip
from datetime import *
from FeatureExtractor import *
import cProfile
import itertools

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

class GridSearch:
    MIN_FEATURE_COUNT=3
    MIN_TWEETS = 5

    MODES=["normal", "baseline"]
    XR_MODES=["lrxr", "lr"]
    ATTRIBUTES=["currentteam"]
    NEGATIVE_SAMPLE_RATE=[1]
    FEATURE_WINDOW_DAYS=[1, 4, 8] # 16, 32
    EXPECTATIONS=[0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20] # 0.15, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01

    NEGATIVE_FILE="../training_data2/negative.gz"
    TRAIN_RANGE=(datetime(year=2008,month=9,day=1), datetime(year=2011,month=6,day=1))
    DEV_RANGE=(datetime(year=2011,month=6,day=5), datetime(year=2012,month=1,day=1))

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
            tweet['y'] = -1
            return

        if mode == "normal":
            tweet['y'] = self.classify_by_timediff(time_since_tweet)
        elif mode == "baseline":
            if in_test_set:
                tweet['y'] = self.classify_by_timediff(time_since_tweet)
            else:
                tweet['y'] = 1
        else:
            assert False

    def __init__(self, attr):
        self.ATTRIBUTES=attr
        self.epTweets = {}
        #self.allTweets = []
        self.trainTweets = []
        self.devTweets   = []
        self.editDates = {}
        self.wpEdits = {}

        labeledIds = set([])

        idEps = set([])

        for f in self.ATTRIBUTES:
            print "reading %s" % f
            for line in open("../training_data2/" + f):
                (label, rel, arg1, arg2, tweetDate, editDate, timeSinceTweet, tweetStr, editStr) = line.strip().split('\t')

                timeSinceTweet = float(timeSinceTweet)

    #            if timeSinceTweet < 0 or timeSinceTweet > 10 * 60 * 24:
    #                continue

                edit = json.loads(editStr)
                title = edit['title']

                tweet = json.loads(tweetStr)
                tweet['arg1']  = arg1
                tweet['arg2']  = arg2
                tweet['title'] = title
                #tweetStr = json.dumps(tweet)

                tweet['datetime'] = ParseDate(tweet['created_at'])

                idEp = tweet['sid'] + '\t' + arg1 + '\t' + arg2
                if idEp in idEps:
                    continue
                idEps.add(idEp)

                ep  = "%s\t%s" % (title,arg2)
                if not self.epTweets.has_key(ep):
                    self.epTweets[ep] = []
                self.epTweets[ep].append(tweet)

                tweet['from_negative'] = False
                tweet['time_since_tweet'] = timeSinceTweet
                #self.classify_tweet("normal", tweet, timeSinceTweet)

                ################################################################################################
                #Keep track of the IDs in the labeled dataset so we can exclude them from the unlabeled data...
                ################################################################################################
                labeledIds.add(tweet['sid'])

                #self.allTweets.append(tweet)
                isInTrainRange = inside_range(self.TRAIN_RANGE, tweet['datetime'])
                isInDevRange   = inside_range(self.DEV_RANGE, tweet['datetime'])
                if isInTrainRange:
                    self.trainTweets.append(tweet)
                elif isInDevRange:
                    self.devTweets.append(tweet)

                if not self.editDates.has_key(ep):
                    self.editDates[ep] = set()
                self.editDates[ep].add(editDate)

                if not self.wpEdits.has_key(ep):
                    self.wpEdits[ep] = set()
                self.wpEdits[ep].add(editStr)

                del tweet['loc']
                del tweet['uid']
                del tweet['sid']
                del tweet['eventTags']
                if 'from_date' in tweet:
                    del tweet['from_date']

        print "done reading positive"
        print "len(trainTweets)=%s" % len(self.trainTweets)
        print "len(devTweets)=%s" % len(self.devTweets)
        #print mode
        print "reading negative"

        nNeg = 0
        idEps = set([])
        for line in gzip.open(self.NEGATIVE_FILE):
            nNeg += 1
            if nNeg % 100000 == 0:
                print "number of negative read: %s" % nNeg

            (arg1, arg2, tweetDate, tweetStr) = line.strip().split('\t')

            tweet = json.loads(tweetStr)
            tweet['arg1'] = arg1
            tweet['title'] = arg1
            tweet['arg2'] = arg2
            tweet['y'] = -1
            tweet['from_negative'] = True
            tweet['time_since_tweet'] = 0
            tweet['datetime'] = ParseDate(tweet['created_at'])

            isInTrainRange = inside_range(self.TRAIN_RANGE, tweet['datetime'])
            isInDevRange   = inside_range(self.DEV_RANGE, tweet['datetime'])

            if not (isInTrainRange or isInDevRange):
                continue

            idEp = tweet['sid'] + '\t' + arg1 + '\t' + arg2
            if idEp in idEps:
                continue
            idEps.add(idEp)

            if not tweet['sid'] in labeledIds:
                if isInTrainRange:
                    self.trainTweets.append(tweet)
                elif isInDevRange:
                    self.devTweets.append(tweet)

            del tweet['loc']
            del tweet['uid']
            del tweet['eventTags']
            if 'from_date' in tweet:
                del tweet['from_date']

        # set([u'loc', 'y', u'uid', 'title', u'eventTags', 'arg1', u'created_at',
        #       'from_date', 'datetime', 'arg2', u'neTags'])
        # fields = set()
        # for tweet in self.trainTweets + self.devTweets:
        #     for key in tweet.iterkeys():
        #         fields.add(key)
        # print fields

        print "done reading negative"
        print "len(trainTweets)=%s" % len(self.trainTweets)
        print "len(devTweets)=%s" % len(self.devTweets)

        self.trainTweets.sort(lambda a,b: cmp(a['datetime'], b['datetime']))
        self.devTweets.sort(lambda a,b: cmp(a['datetime'], b['datetime']))

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

    def MakePredictions(self, model, devData):
        predictions = []
        for fields in devData:
            x = fields[0]
            y = fields[1]
            prediction = model.Predict(x)
            #tweetDate = fields[2]
            tweet = fields[5]
            tweet['datetime'] = ParseDate(tweet['created_at'])
            tweetDate = tweet['created_at']
            arg1 = fields[3]
            arg2 = fields[4]
            ep = "%s\t%s" % (arg1,arg2)
            #print ep
            wpEdits = list(self.wpEdits.get(ep, []))
            editDate = None
            if len(wpEdits) > 0:
                editDate = datetime.fromtimestamp(json.loads(wpEdits[0])['timestamp'])
            predictions.append({'y':y, 'pred':prediction, 'arg1':arg1, 'arg2':arg2, 'tweetDate':tweetDate, 'editDate':editDate, 'tweet':tweet , 'wpEdits':wpEdits})
        return predictions

    def Run(self, output_dir):
        # print "Testing baseline vs normal"
        # self.ClassifyAll("normal")
        # P, D, U = self.PDU(self.trainTweets)
        # print "P (normal)   = %s" % P
        # print "D (normal)   = %s" % D
        # print "U (normal)   = %s" % U
        # self.ClassifyAll("baseline")
        # P, D, U = self.PDU(self.trainTweets)
        # print "P (baseline) = %s" % P
        # print "D (baseline) = %s" % D
        # print "U (baseline) = %s" % U

        self.ClassifyAll("normal")
        attr = ','.join(self.ATTRIBUTES)
        attr_all_file = open(os.path.join(output_dir, "%s.gs" % attr), 'w+')
        for nsr in self.NEGATIVE_SAMPLE_RATE:
            print "Subsampling"
            trainTweets, devTweets = self.Subsample(nsr)

            for fwd in self.FEATURE_WINDOW_DAYS:
                trainData, devData = self.ExtractFeatures(trainTweets, devTweets, fwd)

                for mode, xr in itertools.product(self.MODES, self.XR_MODES):
                    print "Reclassifying"
                    self.ClassifyAll(mode)
                    self.UpdateDataClasses(trainData, devData)
                    P, D, U = self.PDU(self.trainTweets)
                    print "P = %s" % P
                    print "D = %s" % D
                    print "U = %s" % U

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

                        subdir = "%s_%s_%s_%s_%s_%s" % (attr, mode, xr, fwd, nsr, p_ex)
                        mkdir_p(os.path.join(output_dir, subdir))
                        paramOut   = open(os.path.join(output_dir, subdir, 'paramOut'), 'w+')
                        PRout      = open(os.path.join(output_dir, subdir, 'PRout'), 'w+')
                        maxFout    = open(os.path.join(output_dir, subdir, 'maxFout'), 'w+')
                        predOut    = open(os.path.join(output_dir, subdir, 'predOut'), 'w+')

                        paramOut.write("mode                 = %s\n" % mode)
                        paramOut.write("xr_mode              = %s\n" % xr)
                        paramOut.write("feature_window_days  = %s\n" % fwd)
                        paramOut.write("negative_sample_rate = %s\n" % nsr)
                        paramOut.write("p_ex                 = %s\n" % p_ex)
                        paramOut.write("F                    = %s\n" % F)
                        paramOut.close()

                        maxF = 0
                        PR = self.ListPRF(predictions, N)
                        for i in range(len(predictions)):
                            (T, P, R, F) = PR[i]
                            PRout.write("%s\t%s\t%s\t%s\n" % (T, P,R,F))
                            if F > maxF:
                                maxF = F
                        PRout.close()

                        maxFout.write(str(maxF) + "\n")
                        maxFout.close()

                        for p in predictions:
                            dt = p['tweet']['datetime']
                            et = p['editDate']

                            del p['tweet']['datetime']
                            del p['editDate']

                            predOut.write(json.dumps(p) + "\n")

                            p['tweet']['datetime'] = dt
                            p['editDate'] = et
                        predOut.close()

                        model.PrintWeights(os.path.join(output_dir, subdir, 'weights'))

                        attr_all_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (attr, mode, xr, fwd, nsr, p_ex, maxF))
                        attr_all_file.flush()


        attr_all_file.close()

    lastGenDataParams = None
    #PRECONDITION: tweets must be sorted by time before calling GenData
    def GenData(self, tweets, feature_window_days):
        result = []
        prevDates = {}

        for target in tweets:
            ep = "%s\t%s" % (target['title'], target['arg2'])
            if prevDates.has_key(ep) and target['datetime'] - prevDates[ep] < timedelta(minutes=10):
                continue
            prevDates[ep] = target['datetime']

            tweetDate = target['datetime'].strftime("%Y-%m-%d")
            title     = target['title']
            arg2      = target['arg2']
            ep = "%s\t%s" % (title, arg2)
            if ep in self.epTweets and len(self.epTweets[ep]) >= self.MIN_TWEETS:
                fe = FeatureExtractorBinary([x for x in self.epTweets[ep] if x['datetime'] <= target['datetime'] and x['datetime'] > (target['datetime'] - timedelta(days=feature_window_days))], target, title, arg2)
                fe.ComputeBinaryFeatures()
                fe.ComputeTimexFeatures()
#                fe.ComputeEntityFeatures()
#                fe.ComputeVerbFeatures()
                #result.append((fe.Features(), 1))
                result.append([fe.Features(), target['y'], tweetDate, title, arg2, target])
        return result

    def PDU(self, data):
        total = 0
        positive = 0
        discarded = 0
        unknown = 0

        for tweet in data:
            if tweet['y'] == 1:
                positive += 1
            elif tweet['y'] == -1:
                unknown += 1
            else:
                discarded += 1
            total += 1

        P = positive / float(total)
        D = discarded / float(total)
        U = unknown / float(total)

        return (P, D, U)

    def ClassifyAll(self, mode):
        for tweet in self.trainTweets + self.devTweets:
            isInTrainRange = inside_range(self.TRAIN_RANGE, tweet['datetime'])
            isInDevRange   = inside_range(self.DEV_RANGE, tweet['datetime'])
            self.classify_tweet(mode, tweet, tweet['from_negative'], isInDevRange, tweet['time_since_tweet'])

    def Subsample(self, negative_sample_rate):
        self.epTweets = {}
        train = []
        dev   = []

        for tweet in self.trainTweets + self.devTweets:
            isInTrainRange = inside_range(self.TRAIN_RANGE, tweet['datetime'])
            isInDevRange   = inside_range(self.DEV_RANGE, tweet['datetime'])

            arg1 = tweet['arg1']
            arg2 = tweet['arg2']
            title = tweet['title']

            if tweet['from_negative']:
                if (not (hash("%s\t%s" % (arg1, arg2)) % negative_sample_rate == 0)) and isInTrainRange:
                    continue

            epd = "%s\t%s" % (title,arg2)
            if not self.epTweets.has_key(epd):
                self.epTweets[epd] = []
            self.epTweets[epd].append(tweet)

            if isInTrainRange:
                train.append(tweet)
            elif isInDevRange:
                dev.append(tweet)

        return (train, dev)

    def UpdateDataClasses(self, trainData, devData):
        for v in itertools.chain(iter(trainData), iter(devData)):
            # [fe.Features(), target['y'], tweetDate, title, arg2, target]
            target = v[-1]
            v[1] = target['y']

    def ExtractFeatures(self, trainTweets, devTweets, feature_window_days):
        print "Extracting Features"
        trainData = self.GenData(trainTweets, feature_window_days)
        fs = FeatureSelectionEntity(trainTweets, trainData)
        trainData = fs.FilterFeaturesByCount(self.MIN_FEATURE_COUNT)

        devData = self.GenData(devTweets, feature_window_days)
        print "Done Extracting Features"
        print "len(self.train)=%s" % len(trainData)
        print "len(self.dev)=%s" % len(devData)

        return (trainData, devData)

PROFILE=False

if __name__ == "__main__":

    pr = None
    if PROFILE:
        pr = cProfile.Profile()
        pr.enable()

    ds = GridSearch(sys.argv[1].split(','))
    ds.Run(sys.argv[2])

    if PROFILE:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
