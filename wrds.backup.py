import sys
import FeatureExtractor
import LR
import json
import os

sys.path.append('../../weakly_supervised_events/python')

from HDEM import *

def ParseDate(string):
    result = None
    try:
        result = datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        result = datetime.strptime(string, '%a %b %d %H:%M:%S +0000 %Y')    
    return result

class wrds:
    #MIN_FEATURE_COUNT=3
    MIN_FEATURE_COUNT=4
    #MIN_FEATURE_COUNT=10
    
    #POS_MAX_DAYS=10
    #NEG_MIN_DAYS=100

    #POS_MAX_DAYS=3
    #NEG_MIN_DAYS=20

    POS_MAX_DAYS=10
    NEG_MIN_DAYS=20

    #MIN_TWEETS = 5
    #MIN_TWEETS = 3
    #MIN_TWEETS = 1
    #MIN_TWEETS = 2
    MIN_TWEETS = 5
    
    def __init__(self, inFiles, negativeFile=None):
        self.epTweets = {}
        self.allTweets = []
        self.editDates = {}
        self.wpEdits = {}

        print "reading negative"

        if negativeFile != None:
            for line in open(negativeFile):
                (arg1, arg2, tweetDate, tweetStr) = line.strip().split('\t')

                tweet = json.loads(tweetStr)
                tweet['arg1'] = arg1
                tweet['arg2'] = arg2
                tweet['y'] = -1
                tweet['datetime'] = ParseDate(tweet['created_at'])

                epd = "%s\t%s" % (arg1,arg2)
                if not self.epTweets.has_key(epd):
                    self.epTweets[epd] = []
                #self.epTweets[epd].append(json.loads(tweetStr))
                self.epTweets[epd].append(tweet)

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
                if ( timeSinceTweet < (60 * 60 * 24 * self.POS_MAX_DAYS) ) and ( timeSinceTweet > (-60 * 60 * 24 * self.POS_MAX_DAYS) ):
                    tweet['y'] = 1
                elif timeSinceTweet > (60 * 60 * 24 * self.NEG_MIN_DAYS) or timeSinceTweet < -(60 * 60 * 24 * self.NEG_MIN_DAYS):
                    tweet['y'] = -1
                else:
                    tweet['y'] = 0

                self.allTweets.append(tweet)

                if not self.editDates.has_key(ep):
                    self.editDates[ep] = set()
                self.editDates[ep].add(editDate)

                if not self.wpEdits.has_key(ep):
                    self.wpEdits[ep] = set()
                self.wpEdits[ep].add(editStr)

    def Train(self, data):
        #self.model = LRclassifier()
        self.model = LR_XRclassifier()
        #self.model = XR2classifier()
        #self.model = SVMclassifier()
        #self.model.Train(data, l2=100.0)
        #self.model.Train([x for x in data if x[1] != 0], l2=100.0)
        self.model.Train([x for x in data if x[1] != 0], l2=1.0)

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

    def FilterFirstPredictions(self, predictions):
        fps = {}
        for p in predictions:
            arg1 = p['arg1']
            arg2 = p['arg2']
            dt   = p['tweet']['datetime']

            key = "%s\t%s" % (arg1,arg2)
            if (not fps.has_key(key)) or fps[key]['tweet']['datetime'] > dt:
                fps[key] = p
        return fps.values()

    def Test2(self, data, outDir):
        """ Only evaluate the first prediction for each entity pair """
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

        filteredPredictions = self.FilterFirstPredictions(predictions)
        N = len([x for x in filteredPredictions if x['y'] == 1])
        print N
        maxF = 0.0
        predictions.sort(lambda a,b: cmp(b['pred'], a['pred']))
        for i in range(1, len(predictions)+1):
            pred_thresh = self.FilterFirstPredictions(predictions[0:i])
            (P, R, F) = self.PointPRF(pred_thresh, N)
            PRout.write("%s\t%s\t%s\n" % (P,R,F))
            if F > maxF:
                maxF = F
            #predOut.write(str(p) + "\n")

        predictions.sort(lambda a,b: cmp(b['pred'], a['pred']))
        for p in predictions:
            if p['tweet'].has_key('datetime'):
                del p['tweet']['datetime']
            predOut.write(json.dumps(p) + "\n")

        maxFout.write(str(maxF) + "\n")

        PRout.close()
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

    def GenData(self, tweets):
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
            if len(self.epTweets[ep]) >= self.MIN_TWEETS:
                fe = FeatureExtractorBinary([x for x in self.epTweets[ep] if x['datetime'] <= target['datetime'] and x['datetime'] > (target['datetime'] - timedelta(days=1))], target)
                fe.ComputeBinaryFeatures()
                fe.ComputeTimexFeatures()
#                fe.ComputeEntityFeatures()
#                fe.ComputeVerbFeatures()
                #result.append((fe.Features(), 1))
                result.append((fe.Features(), target['y'], tweetDate, title, arg2, target))
        return result

    def SplitTrainDevTest(self, trainP, devP, testP):
        if trainP + devP + testP != 1.0:
            raise Exception('train/dev/test proportions do not sum to 1') 

        self.allTweets.sort(lambda a,b: cmp(a['datetime'], b['datetime']))
        
        trainSize = int(len(self.allTweets)*trainP)
        devSize   = int(len(self.allTweets)*devP)
        testSize  = int(len(self.allTweets)*testP)

        print "trainSize: %s devSize: %s testSize: %s" % (trainSize, devSize, testSize)

        train = self.allTweets[0:trainSize]
        dev   = self.allTweets[trainSize:trainSize+devSize]
        #dev   = train
        test  = self.allTweets[trainSize+devSize:]

        print "Extracting Features"

        self.train = self.GenData(train)
        fs = FeatureSelectionEntity(train, self.train)
        self.train = fs.FilterFeaturesByCount(self.MIN_FEATURE_COUNT)

        self.dev = self.GenData(dev)

        self.test = self.GenData(test)

        print "Done Extracting Features"

if __name__ == "__main__":
    ds = wrds(sys.argv[1].split(','), sys.argv[2])
    #ds = wrds(sys.argv[1].split(','))

    ds.SplitTrainDevTest(0.5,0.25,0.25)
    #ds.SplitTrainDevTest(1.0, 0.0, 0.0)
    #ds.SplitTrainDevTest(0.5,0.5,0.0)

    print len(ds.train)
    print len(ds.dev)
    print len(ds.test)

    ds.Train(ds.train)

    if not os.path.isdir('../experiments/%s' % sys.argv[1]):
        os.makedirs('../experiments/%s' % sys.argv[1])
    #ds.Test(ds.dev, '../experiments/%s' % sys.argv[1])
    ds.Test2(ds.dev, '../experiments/%s' % sys.argv[1])
