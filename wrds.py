import sys
import LR
import json
import os
import gzip
from datetime import *
from FeatureExtractor import *
import cProfile

#sys.path.append('../../weakly_supervised_events/python')

#from HDEM import *
from Classifier import *
from Vocab import *

def ParseDate(string):
    result = None
    try:
        result = datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        result = datetime.strptime(string, '%a %b %d %H:%M:%S +0000 %Y')    
    return result

def inside_range(range, x):
    return range[0] <= x and x <= range[1]

class wrds:
    MIN_FEATURE_COUNT=3
    #MIN_FEATURE_COUNT=4
    #MIN_FEATURE_COUNT=10
    
    #POS_MAX_DAYS=10
    #NEG_MIN_DAYS=100

    #POS_MAX_DAYS=3
    #NEG_MIN_DAYS=20

    #POS_MAX_DAYS=10
    #NEG_MIN_DAYS=20

    #POS_MAX_DAYS=20
    #NEG_MIN_DAYS=40
    #NEG_MIN_DAYS=100
    #POS_MAX_DAYS=60
    POS_MAX_DAYS=10
    #NEG_MIN_DAYS=100
    NEG_MIN_DAYS=30

    #MIN_TWEETS = 5
    #MIN_TWEETS = 3
    #MIN_TWEETS = 1
    #MIN_TWEETS = 2
    MIN_TWEETS = 5

    FEATURE_WINDOW_DAYS=1

    NEGATIVE_SAMPLE_RATE=1

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

    #Time range for train / dev test split are given as arguments
    def __init__(self, inFiles, negativeFile=None, trainRange=(datetime(year=2008,month=9,day=1), datetime(year=2011,month=6,day=1)), 
                                                   devRange=(datetime(year=2011,month=6,day=5), datetime(year=2011,month=9,day=1)),
                                                   #testRange=(datetime(year=2011,month=9,day=5), datetime(year=2012,month=1,day=1))):
                                                   testRange=(datetime(year=2011,month=9,day=5), datetime(year=2011,month=10,day=1)),
                                                   mode="normal"):
                                                   #devRange=(datetime(year=2014,month=1,day=5), datetime(year=2014,month=5,day=27))):
        self.epTweets = {}
        #self.allTweets = []
        self.trainTweets = []
        self.devTweets   = []
        self.testTweets  = []
        self.editDates = {}
        self.wpEdits = {}

        labeledIds = set([])

        for f in inFiles:
            print "reading %s" % f
            for line in open(f):
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
                tweetStr = json.dumps(tweet)

                tweet['datetime'] = ParseDate(tweet['created_at'])

                ep  = "%s\t%s" % (title,arg2)
                if not self.epTweets.has_key(ep):
                    self.epTweets[ep] = []
                self.epTweets[ep].append(tweet)

                ##############################################################
                # Positive: Tweet is += x days from edit date
                # Negative: Tweet is > y days before or after edit date
                ##############################################################
                # if mode == "normal":
                #     #if ( timeSinceTweet < (60 * 60 * 24 * self.POS_MAX_DAYS) ) and ( timeSinceTweet > (-60 * 60 * 24 * self.POS_MAX_DAYS) ):
                #     if timeSinceTweet > -3*(60 * 60 * 24) and timeSinceTweet < (60 * 60 * 24 * self.POS_MAX_DAYS):
                #             tweet['y'] = 1
                #     #elif timeSinceTweet > (60 * 60 * 24 * self.NEG_MIN_DAYS) or timeSinceTweet < -(60 * 60 * 24 * self.NEG_MIN_DAYS):
                #     elif timeSinceTweet < 0 and timeSinceTweet < -(60 * 60 * 24 * self.NEG_MIN_DAYS):
                #             tweet['y'] = -1
                #     else:
                #             tweet['y'] = 0
                # else:
                #     tweet['y'] = 1
                isInDevRange   = inside_range(devRange, tweet['datetime'])
                self.classify_tweet(mode, tweet, False, isInDevRange, timeSinceTweet)
                
                ################################################################################################
                #Keep track of the IDs in the labeled dataset so we can exclude them from the unlabeled data...
                ################################################################################################
                labeledIds.add(tweet['sid'])

                #self.allTweets.append(tweet)
                if   tweet['datetime'] >= trainRange[0] and tweet['datetime'] <= trainRange[1]:
                    self.trainTweets.append(tweet)
                elif tweet['datetime'] >= devRange[0] and tweet['datetime'] <= devRange[1]:
                    self.devTweets.append(tweet)
                elif tweet['datetime'] >= testRange[0] and tweet['datetime'] <= testRange[1]:
                    self.testTweets.append(tweet)

                if not self.editDates.has_key(ep):
                    self.editDates[ep] = set()
                self.editDates[ep].add(editDate)

                if not self.wpEdits.has_key(ep):
                    self.wpEdits[ep] = set()
                self.wpEdits[ep].add(editStr)

        print "done reading positive"
        print "len(trainTweets)=%s" % len(self.trainTweets)
        print "len(devTweets)=%s" % len(self.devTweets)
        print "len(testTweets)=%s" % len(self.testTweets)
        print mode
        print "reading negative"

        nNeg = 0
        if negativeFile != None:
            #for line in open(negativeFile):
            for line in gzip.open(negativeFile):
                nNeg += 1
                if nNeg % 100000 == 0:
                    print "number of negative read: %s" % nNeg

                (arg1, arg2, tweetDate, tweetStr) = line.strip().split('\t')

                tweet = json.loads(tweetStr)
                tweet['arg1'] = arg1
                tweet['title'] = arg1
                tweet['arg2'] = arg2
                # tweet['y'] = -1
                tweet['datetime'] = ParseDate(tweet['created_at'])

                isInTrainRange = tweet['datetime'] >= trainRange[0] and tweet['datetime'] <= trainRange[1]
                isInDevRange   = tweet['datetime'] >= devRange[0] and tweet['datetime'] <= devRange[1]
                isInTestRange  = tweet['datetime'] >= testRange[0] and tweet['datetime'] <= testRange[1]

                self.classify_tweet(mode, tweet, True, isInDevRange, 0)

                if not (isInTrainRange or isInDevRange or isInTestRange):
                    continue

                ##########################################################
                # Randomly sample 1/NEGATIVE_SAMPLE_RATE entity paris
                ##########################################################
                #if (not (hash("%s\t%s" % (arg1,arg2)) % self.NEGATIVE_SAMPLE_RATE == 0)) and tweet['datetime'] >= trainRange[0] and tweet['datetime'] <= trainRange[1]:
                if (not (hash("%s\t%s" % (arg1,arg2)) % self.NEGATIVE_SAMPLE_RATE == 0)) and isInTrainRange:
                    continue

                if not tweet['sid'] in labeledIds:
                    epd = "%s\t%s" % (arg1,arg2)
                    if not self.epTweets.has_key(epd):
                        self.epTweets[epd] = []
                    #self.epTweets[epd].append(json.loads(tweetStr))
                    self.epTweets[epd].append(tweet)

                    #self.allTweets.append(tweet)
                    if   isInTrainRange:
                        self.trainTweets.append(tweet)
                    elif isInDevRange:
                        self.devTweets.append(tweet)
                    elif isInTestRange:
                        self.testTweets.append(tweet)

        print "done reading negative"
        print "len(trainTweets)=%s" % len(self.trainTweets)
        print "len(devTweets)=%s" % len(self.devTweets)
        print "len(testTweets)=%s" % len(self.testTweets)
        print "len(positive trainTweets)=%s" % sum(1 for x in self.trainTweets if x['y'] == 1)
        print "len(positive devTweets)=%s" % sum(1 for x in self.devTweets if x['y'] == 1)

    def Train(self, data):
        #self.model = LRclassifier()
        self.model = LR_XRclassifier()
        #self.model = XR2classifier()
        #self.model = SVMclassifier()
        #self.model.Train(data, l2=100.0)
        self.model.Train([x for x in data if x[1] != 0], l2=100.0)
        #self.model.Train([x for x in data if x[1] != 0], l2=1.0)
        print "n features: %s" % self.model.vocab.GetVocabSize()

    def PointPRF(self, predictions, N):
        """ Computes a single precision and recall of a list of predictions """
        tp = 0.0
        fp = 0.0
        fn = 0.0

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

        return (P,R,F)

    def ListPRF(self, predictions, N):
        """ Computes a single precision and recall of a list of predictions """
        tp = 0.0
        fp = 0.0
        fn = 0.0
        result = []

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

            result.append((P,R,F))
        return result


    def FilterFirstPredictions(self, predictions):
        fps = {}
        for p in predictions:
            arg1 = p['arg1']
            arg2 = p['arg2']
            dt   = p['tweet']['datetime']
            pred = p['pred']

            key = "%s\t%s" % (arg1,arg2)
            #NOTE: trying highest scoring prediction for each entity pair rather than first above threshold... I don't know, I think first above threshold makes more sense...
            #if (not fps.has_key(key)) or fps[key]['pred'] < pred:
            if (not fps.has_key(key)) or fps[key]['tweet']['datetime'] > dt:
                fps[key] = p
        return fps.values()

    def AverageLag(self, predictions):
        lsum = 0.0
        ltot = 0.0
        for p in predictions:
            if p['y'] == 1:
                lsum += (p['editDate'] - p['tweet']['datetime']).total_seconds()
                ltot += 1.0
        if ltot > 0:
            return lsum / ltot
        else:
            return 0.0

    def Test2(self, data, outDir, computePL=True):
        """ Only evaluate the first prediction for each entity pair """
        PRout   = open("%s/PRout"   % outDir, 'w+')
        PLout   = open("%s/PLout"   % outDir, 'w+')
        maxFout = open("%s/maxFout" % outDir, 'w+')
        predOut = open("%s/predOut" % outDir, 'w+')
        self.model.PrintWeights("%s/weights" % outDir)
        
        predictions = []
        for fields in data:
            x = fields[0]
            y = fields[1]
            prediction = self.model.Predict(x)
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
            #predictions.append({'x':x, 'y':y, 'pred':prediction, 'arg1':arg1, 'arg2':arg2, 'tweetDate':tweetDate, 'tweet':tweet , 'wpEdits':wpEdits})
            #predictions.append({'y':y, 'patterns':[f for f in x.keys() if 'arg1' in f], 'pred':prediction, 'arg1':arg1, 'arg2':arg2, 'tweetDate':tweetDate, 'editDate':editDate, 'tweet':tweet , 'wpEdits':wpEdits})
            predictions.append({'y':y, 'pred':prediction, 'arg1':arg1, 'arg2':arg2, 'tweetDate':tweetDate, 'editDate':editDate, 'tweet':tweet , 'wpEdits':wpEdits})


        ###############################################
        #NOTE: Recall is calculated based on # of 
        # entity pairs edited after the date of the
        # first tweet in the test set
        tweetDates = [x['tweet']['datetime'] for x in predictions]
        tweetDates.sort()
        firstTweetDate = tweetDates[0]
        lastTweetDate = tweetDates[-1]
        print str(firstTweetDate)
        #print predictions
        #N = len(list(set([(x['arg1'], x['arg2']) for x in predictions if x['y'] == 1 and x['editDate'] > firstTweetDate and x['editDate'] < lastTweetDate])))
        #N = len(list(set([(x['arg1'], x['arg2']) for x in predictions if x['y'] == 1])))
        N = sum(1 for x in predictions if x['y'] == 1)
        print "N=%s" % N
        ###############################################

        maxF = 0.0
        predictions.sort(lambda a,b: cmp(b['pred'], a['pred']))
        (prevP, prevR) = (0, 0)

        if computePL:
            for i in range(1, len(predictions)+1):
                pred_thresh = self.FilterFirstPredictions(predictions[0:i])
                if len(pred_thresh) == 0:
                    continue
                (P, R, F) = self.PointPRF(pred_thresh, N)
                if (P,R) == (prevP, prevR):
                    continue
                (prevP, prevR) = (P,R)

                PLout.write("%s\t%s\n" % (P, self.AverageLag(pred_thresh)))
                PRout.write("%s\t%s\t%s\n" % (P,R,F))
                if F > maxF:
                    maxF = F
        else:
            PR = self.ListPRF(predictions, N)
            for i in range(len(predictions)):
                (P, R, F) = PR[i]
                PRout.write("%s\t%s\t%s\n" % (P,R,F))
                if F > maxF:
                    maxF = F

        #predictions.sort(lambda a,b: cmp(b['pred'], a['pred']))
        for p in predictions:
            if p['tweet'].has_key('datetime'):
                del p['tweet']['datetime']
            del p['editDate']
            predOut.write(json.dumps(p) + "\n")

        maxFout.write(str(maxF) + "\n")

        PRout.close()
        PLout.close()
        predOut.close()
        maxFout.close()
        

    def Test(self, data, outDir):
        PRout   = open("%s/PRout"   % outDir, 'w+')
        maxFout = open("%s/maxFout" % outDir, 'w+')
        predOut = open("%s/predOut" % outDir, 'w+')
        self.model.PrintWeights("%s/weights" % outDir)
        
        predictions = []
        for fields in data:
            x = fields[0]
            y = fields[1]
            prediction = self.model.Predict(x)
            #tweetDate = fields[2]
            tweet = fields[5]
            tweetDate = tweet['created_at']
            arg1 = fields[3]
            arg2 = fields[4]
            ep = "%s\t%s" % (arg1,arg2)
            #print ep
            wpEdits = list(self.wpEdits.get(ep, []))
            #predictions.append({'x':x, 'y':y, 'pred':prediction, 'arg1':arg1, 'arg2':arg2, 'tweetDate':tweetDate, 'tweet':tweet , 'wpEdits':wpEdits})
            predictions.append({'y':y, 'patterns':[f for f in x.keys() if 'arg1' in f], 'pred':prediction, 'arg1':arg1, 'arg2':arg2, 'tweetDate':tweetDate, 'tweet':tweet , 'wpEdits':wpEdits})
            
        tp = 0.0
        fp = 0.0
        fn = 0.0
        N  = len([x for x in predictions if x['y'] == 1])
        maxF = 0.0

        predictions.sort(lambda a,b: cmp(b['pred'], a['pred']))
        for p in predictions:
            if p['y'] == 1:
                tp += 1
            elif p['y'] == -1:
                fp += 1
            fn = N - tp
            P = tp / (tp + fp)
            R = 0.0
            if tp + fn > 0:
                R = tp / (tp + fn)
            if P == 0 or R == 0:
                F = 0
            else:
                F = 2 * P * R / (P + R)

            if F > maxF:
                maxF = F

            PRout.write("%s\t%s\t%s\n" % (P,R,F))
            #predOut.write(str(p) + "\n")
            if p['tweet'].has_key('datetime'):
                del p['tweet']['datetime']
            predOut.write(json.dumps(p) + "\n")

        maxFout.write(str(maxF) + "\n")

        PRout.close()
        predOut.close()
        maxFout.close()

    #PRECONDITION: tweets must be sorted by time before calling GenData
    def GenData(self, tweets):
        result = []
        prevDates = {}
        for target in tweets:

            ep = "%s\t%s" % (target['title'], target['arg2'])
            if prevDates.has_key(ep) and target['datetime'] - prevDates[ep] < timedelta(minutes=10):
            #if prevDates.has_key(ep) and target['datetime'] - prevDates[ep] < timedelta(minutes=30):
                continue
            prevDates[ep] = target['datetime']
            
            tweetDate = target['datetime'].strftime("%Y-%m-%d")
            title     = target['title']
            arg2      = target['arg2']
            ep = "%s\t%s" % (title, arg2)
            if len(self.epTweets[ep]) >= self.MIN_TWEETS:
                ###########################################
                # Feature window is previous 24 hours
                ###########################################
                #fe = FeatureExtractorBinary([x for x in self.epTweets[ep] if x['datetime'] <= target['datetime'] and x['datetime'] > (target['datetime'] - timedelta(days=1))], target, title, arg2)
                fe = FeatureExtractorBinary([x for x in self.epTweets[ep] if x['datetime'] <= target['datetime'] and x['datetime'] > (target['datetime'] - timedelta(days=self.FEATURE_WINDOW_DAYS))], target, title, arg2)
                fe.ComputeBinaryFeatures()
                fe.ComputeTimexFeatures()
#                fe.ComputeEntityFeatures()
#                fe.ComputeVerbFeatures()
                #result.append((fe.Features(), 1))
                result.append((fe.Features(), target['y'], tweetDate, title, arg2, target))
        return result

    #def SplitTrainDevTest(self, trainP, devP, testP):
    def ExtractFeatures(self):
        #self.allTweets.sort(lambda a,b: cmp(a['datetime'], b['datetime']))
        self.trainTweets.sort(lambda a,b: cmp(a['datetime'], b['datetime']))
        self.devTweets.sort(lambda a,b: cmp(a['datetime'], b['datetime']))
        self.testTweets.sort(lambda a,b: cmp(a['datetime'], b['datetime']))
        
#        trainSize = int(len(self.allTweets)*trainP)
#        devSize   = int(len(self.allTweets)*devP)
#        testSize  = int(len(self.allTweets)*testP)
#
#        print "trainSize: %s devSize: %s testSize: %s" % (trainSize, devSize, testSize)
#
#        train = self.allTweets[0:trainSize]
#        dev   = self.allTweets[trainSize:trainSize+devSize]
#        #dev   = train
#        test  = self.allTweets[trainSize+devSize:]

        train = self.trainTweets
        dev   = self.devTweets
        test  = self.testTweets

        print "Extracting Features"

        self.train = self.GenData(train)
        fs = FeatureSelectionEntity(train, self.train)
        self.train = fs.FilterFeaturesByCount(self.MIN_FEATURE_COUNT)

        self.dev = self.GenData(dev)

        self.test = self.GenData(test)

        print "Done Extracting Features"

#    def TrainTestAll(self):
#        self.allTweets.sort(lambda a,b: cmp(a['datetime'], b['datetime']))
#
#        train = self.allTweets
#
#        print "Extracting Features"
#
#        self.train = self.GenData(train)
#        fs = FeatureSelectionEntity(train, self.train)
#        self.train = fs.FilterFeaturesByCount(self.MIN_FEATURE_COUNT)
#        self.dev = self.GenData(self.allTweets)
#
#        self.test = self.dev
#
#        print "Done Extracting Features"

#PROFILE=True
PROFILE=False

if __name__ == "__main__":

    pr = None
    if PROFILE:
        pr = cProfile.Profile()
        pr.enable()

    ds = wrds(sys.argv[1].split(','), sys.argv[2], mode=sys.argv[4])
    #ds = wrds(sys.argv[1].split(','))

    ds.ExtractFeatures()
    #ds.TrainTestAll()

    print len(ds.train)
    print len(ds.dev)
    print len(ds.test)

    ds.Train(ds.train)


    #####################################################
    # Train
    #####################################################
#    if not os.path.isdir('../%s/%s/train' % (sys.argv[3], sys.argv[1])):
#        os.makedirs('../%s/%s/train' % (sys.argv[3], sys.argv[1]))
#    ds.Test2(ds.train, '../%s/%s/train' % (sys.argv[3], sys.argv[1]), computePL=False)

    #####################################################
    # Dev
    #####################################################
    if not os.path.isdir('../%s/%s/dev' % (sys.argv[3], sys.argv[1])):
        print '../%s/%s/dev' % (sys.argv[3], sys.argv[1])
    os.makedirs('../%s/%s/dev' % (sys.argv[3], sys.argv[1]))
    ds.Test2(ds.dev, '../%s/%s/dev' % (sys.argv[3], sys.argv[1]), computePL=False)

    #####################################################
    # Test
    #####################################################
    if not os.path.isdir('../%s/%s/test' % (sys.argv[3], sys.argv[1])):
        os.makedirs('../%s/%s/test' % (sys.argv[3], sys.argv[1]))
    ds.Test2(ds.test, '../%s/%s/test' % (sys.argv[3], sys.argv[1]), computePL=False)

    if PROFILE:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
