import sys
import FeatureExtractor
import LR
import json
import os

sys.path.append('../../weakly_supervised_events/python')

from HDEM import *

class wrds:
    MIN_FEATURE_COUNT=3
    #MIN_FEATURE_COUNT=10
    
    POS_MAX_DAYS=10
    NEG_MIN_DAYS=100

    #MIN_TWEETS = 5
    #MIN_TWEETS = 3
    #MIN_TWEETS = 1
    #MIN_TWEETS = 2
    MIN_TWEETS = 5
    
    def __init__(self, inFiles, negativeFile=None):
        self.posTweets = {}
        self.negTweets = {}
        self.unkTweets = {}
        self.editDates = {}
        self.wpEdits = {}

        print "reading negative"

        if negativeFile != None:
            for line in open(negativeFile):
                (arg1, arg2, tweetDate, tweetStr) = line.strip().split('\t')

                tweet = json.loads(tweetStr)
                tweet['arg1'] = arg1
                tweet['arg2'] = arg2
                tweetStr = json.dumps(tweet)

                epd = "%s\t%s\t%s" % (tweetDate,arg1,arg2)
                if not self.negTweets.has_key(epd):
                    self.negTweets[epd] = []
                self.negTweets[epd].append(json.loads(tweetStr))

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
                tweet['arg1'] = arg1
                tweet['arg2'] = arg2
                tweetStr = json.dumps(tweet)

                #epd = "%s\t%s\t%s" % (tweetDate,arg1,arg2)
                #ep  = "%s\t%s" % (arg1,arg2)
                epd = "%s\t%s\t%s" % (tweetDate,title,arg2)
                ep  = "%s\t%s" % (title,arg2)

                ##############################################################
                # Positive: Tweet is += 10 days from edit date
                # Negative: Tweet is > 100 days before or after edit date
                ##############################################################
    #            if label == '+':
                if ( timeSinceTweet < (60 * 60 * 24 * self.POS_MAX_DAYS) ) and ( timeSinceTweet > (-60 * 60 * 24 * self.POS_MAX_DAYS) ):
                    if not self.posTweets.has_key(epd):
                        self.posTweets[epd] = []
                    self.posTweets[epd].append(json.loads(tweetStr))
    #            elif label == '-':
                elif timeSinceTweet > (60 * 60 * 24 * self.NEG_MIN_DAYS) or timeSinceTweet < -(60 * 60 * 24 * self.NEG_MIN_DAYS):
                    if not self.negTweets.has_key(epd):
                        self.negTweets[epd] = []
                    self.negTweets[epd].append(json.loads(tweetStr))
                else:
                    if not self.unkTweets.has_key(epd):
                        self.unkTweets[epd] = []
                    self.unkTweets[epd].append(json.loads(tweetStr))

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
        self.model.Train([x for x in data if x[1] != 0], l2=100.0)

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
            tweetDate = fields[2]
            arg1 = fields[3]
            arg2 = fields[4]
            tweet = fields[5]
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
            predOut.write(json.dumps(p) + "\n")

        maxFout.write(str(maxF) + "\n")

        PRout.close()
        predOut.close()
        maxFout.close()

    def GenData(self, epds):
        result = []
        for epd in epds:
            if self.posTweets.has_key(epd) and len(self.posTweets[epd]) >= self.MIN_TWEETS:
                (tweetDate, title, arg2) = epd.split("\t")
                fe = FeatureExtractorBinary(self.posTweets[epd])
                fe.ComputeBinaryFeatures()
#                fe.ComputeEntityFeatures()
                fe.ComputeVerbFeatures()
                #result.append((fe.Features(), 1))
                result.append((fe.Features(), 1, tweetDate, title, arg2, self.posTweets[epd]))
            if self.negTweets.has_key(epd) and len(self.negTweets[epd]) >= self.MIN_TWEETS:
                (tweetDate, title, arg2) = epd.split("\t")
                fe = FeatureExtractorBinary(self.negTweets[epd])
                fe.ComputeBinaryFeatures()
#                fe.ComputeEntityFeatures()
                fe.ComputeVerbFeatures()
                #result.append((fe.Features(), -1))
                result.append((fe.Features(), -1, tweetDate, title, arg2, self.negTweets[epd]))
            if self.unkTweets.has_key(epd) and len(self.unkTweets[epd]) >= self.MIN_TWEETS:
                (tweetDate, title, arg2) = epd.split("\t")
                fe = FeatureExtractorBinary(self.unkTweets[epd])
                fe.ComputeBinaryFeatures()
#                fe.ComputeEntityFeatures()
                fe.ComputeVerbFeatures()
                #result.append((fe.Features(), 0))
                result.append((fe.Features(), 0, tweetDate, title, arg2, self.unkTweets[epd]))
        return result

    def SplitTrainDevTest(self, trainP, devP, testP):
        if trainP + devP + testP != 1.0:
            raise Exception('train/dev/test proportions do not sum to 1') 

        epds = sorted(list(set(self.posTweets.keys() + self.negTweets.keys())))
        
        trainSize = int(len(epds)*trainP)
        devSize   = int(len(epds)*devP)
        testSize  = int(len(epds)*testP)

        train = epds[0:trainSize]
        dev   = epds[trainSize:trainSize+devSize]
        #dev   = train
        test  = epds[trainSize+devSize:]

        self.train = self.GenData(train)
        fs = FeatureSelection(self.train)
        self.train = fs.FilterFeaturesByCount(self.MIN_FEATURE_COUNT)

        self.dev = self.GenData(dev)

        self.test = self.GenData(test)

if __name__ == "__main__":
    ds = wrds(sys.argv[1].split(','), sys.argv[2])
    #ds = wrds(sys.argv[1].split(','))

    ds.SplitTrainDevTest(0.5,0.25,0.25)
    #ds.SplitTrainDevTest(1.0, 0.0, 0.0)
    #ds.SplitTrainDevTest(0.5,0.5,0.0)

    for epd in ds.posTweets.keys():
        (tweetDate, arg1, arg2) = epd.split("\t")
        ep = "%s\t%s" % (arg1, arg2)
        #print str(len(ds.posTweets[epd])) + "\t" + epd + "\t+\t" + ds.posTweets[epd][0]['words'] + "\t" + str(ds.editDates[ep])

    for epd in ds.negTweets.keys():
        (tweetDate, arg1, arg2) = epd.split("\t")
        ep = "%s\t%s" % (arg1, arg2)
        #print str(len(ds.negTweets[epd])) + "\t" + epd + "\t-\t" + ds.negTweets[epd][0]['words'] + "\t" + str(ds.editDates.get(ep, None))

    print len(ds.train)
    print len(ds.dev)
    print len(ds.test)

    ds.Train(ds.train)

    if not os.path.isdir('../experiments/%s' % sys.argv[1]):
        os.makedirs('../experiments/%s' % sys.argv[1])
    ds.Test(ds.dev, '../experiments/%s' % sys.argv[1])
