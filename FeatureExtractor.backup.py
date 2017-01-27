import os
import math
import sys
sys.path.append('/home/rittera/twitter_utils/python')
import TwitterUtils
from datetime import *

import re

BASE_DIR = os.getenv("DSUP_EVENT_DIR")

def isAscii(s):
    try:
        s.decode('ascii')
    except Exception:
        return False
    return True

stop_words = set()
#for line in open('/home/rittera/repos/weakly_supervised_events/python/stop_words'):
for line in open('%s/data/stop_words' % BASE_DIR):
    stop_words.add(line.strip())

#From data collection code.  Don't use these as features
#track_list = set(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
#                  'mon', 'tues', 'wed', 'thur', 'thurs', 'fri', 'sat', 'sun',
#                  'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
#                  'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
#                  '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th', '13th', '14th', '15th', '16th', '17th', '18th', '19th', '20th', '21st', '22nd', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th', '31st',
#                  'tomorrow'])
track_list = set(['ddos', 'tangodown', 'dos', 'tango'])

def GetSegments(words, annotations, tag, lower=True, getIndices=False):
    results = []
    #words = words.split(' ')
    #annotations = annotations.split(' ')

    annotations = [x.split(":")[0] for x in annotations]

    start = None
    for i in range(len(words)):
        if annotations[i].split(':')[0] == 'B-%s' % tag:
            if start != None:
                if getIndices:
                    results.append((' '.join(words[start:i]), (start,i)))
                else:
                    results.append(' '.join(words[start:i]))
            start = i
        elif annotations[i] == 'O' and start != None:
            if getIndices:
                results.append((' '.join(words[start:i]), (start,i)))
            else:
                results.append(' '.join(words[start:i]))
            start = None
    if start != None:
        if getIndices:
            results.append((' '.join(words[start:]), (start,i)))
        else:
            results.append(' '.join(words[start:]))

    if lower:
        if getIndices:
            results = [(x[0].lower(),x[1]) for x in results]
        else:
            results = [x.lower() for x in results]            
    return results

nell_categories = {}
for line in open('../data/nell_categories_simple'):
    fields = line.strip().split('\t')
    if len(fields) != 2:
        continue
    (entity, category) = fields
    entity = entity.strip()
    category = category.strip()
    if not nell_categories.has_key(entity):
        nell_categories[entity] = []
    nell_categories[entity].append(category)

def GetCategory(entity):
    result = []
    if not nell_categories.has_key(entity):
        return 'None'
    lookup = entity
    while nell_categories.has_key(lookup):
        nc = nell_categories[lookup][0]         #TODO: handle more than one category.  This is probably fine for now...
        if nc == 'agent' or nc == 'abstractthing' or nc == 'organization':
            return lookup
        lookup = nell_categories[lookup][0]
    return 'None'

def GetNV(words, pos):
    result = []
    for i in range(len(words)):
        if pos[i][0] == 'N' or pos[i][0] == 'V':
            result.append(words[i])
        else:
            result.append(pos[i])
    return " ".join(result)

class FeatureExtractorBinary:
    def __init__(self, tweets, target, keyword=None):
        self.tweets = tweets
        self.target = target
        for js in tweets:
            #js['preprocessed'] = TwitterUtils.preprocess(' '.join(js['words']))
            js['preprocessed'] = TwitterUtils.preprocess(js['words'])
        self.features = {}
        self.keyword = keyword

    def ComputeEntityFeatures(self):
#        for a1cat in nell_categories.get(self.arg1, []):
#            self.SetFeatureTrue("arg1=%s" % a1cat)
        self.SetFeatureTrue("arg1=%s" % GetCategory(self.arg1))

#        for a2cat in nell_categories.get(self.arg2, []):
#            self.SetFeatureTrue("arg2=%s" % a2cat)
        self.SetFeatureTrue("arg2=%s" % GetCategory(self.arg2))

    def ComputeTimexFeatures(self):
        for tweet in self.tweets:
            timeRef = None

            try:
                timeRef = datetime.strptime(tweet['date'], '%Y%m%d')
            except:
                pass
            try:
                timeRef = datetime.strptime(tweet['date'], '%Y%m')
            except:
                pass
            try:
                timeRef = datetime.strptime(tweet['date'], '%Y')
            except:
                pass

            if timeRef == None:
                continue
            
            if timeRef > tweet['datetime'] + timedelta(days=1):
                self.SetFeatureTrue('FUTURE_TIME_REF')
            if timeRef < tweet['datetime'] - timedelta(days=1):
                self.SetFeatureTrue('PAST_TIME_REF')
            if timeRef < tweet['datetime'] - timedelta(days=365):
                self.SetFeatureTrue('VERY_OLD_TIME_REF')
            if timeRef > tweet['datetime'] + timedelta(days=30):
                self.SetFeatureTrue('FAR_FUTURE_TIME_REF')

    def ComputeBinaryFeatures(self):
        for tweet in self.tweets:
            tokenized = tweet['words'].lower().split(' ')
            neTags    = tweet['neTags'].split(' ')
            pos       = tweet['pos'].split(' ')
            arg1      = tweet['arg1']
            arg2      = tweet['arg2']

            entityIndices = GetSegments(tokenized, neTags, 'ENTITY', getIndices=True)

            a1indices = []
            a2indices = []
            for ei in entityIndices:
                if ei[0] == arg1:
                    a1indices.append(ei[1])
                if ei[0] == arg2:
                    a2indices.append(ei[1])

            for a1 in a1indices:
                for a2 in a2indices:
                    if a1[0] == a2[0]:
                        continue

                    ##########################################################
                    # Words between (example: u'arg2 secretly wed arg1')
                    ##########################################################
                    MAX_WORDS_BETWEEN = 5

                    for k in [1,2,3]:           #k = context size (number of words)
                        feature1 = None
                        feature2 = None
                        if a2[0] - a1[1] > 0:
                            left_context          = " ".join(tokenized[(a1[0]-k):a1[0]])
                            between_context       = " ".join(tokenized[(a1[1]):a2[0]])
                            between_context_short = between_context
                            if a2[0] - a1[1] > 2 * k:
                                between_context_short = " ".join(tokenized[(a1[1]):a1[1]+k]) + "..." + " ".join(tokenized[(a2[0]-k):a2[0]])
                            right_context         = " ".join(tokenized[(a2[1]):(a2[1]+k)])

                            if a2[0] - a1[1] <= MAX_WORDS_BETWEEN:
                                self.SetFeatureTrue("arg1 %s arg2" % between_context)
                                self.SetFeatureTrue("%s arg1 %s arg2" % (left_context, between_context))
                                self.SetFeatureTrue("arg1 %s arg2 %s" % (between_context, right_context))

                            self.SetFeatureTrue("arg1 %s arg2" % between_context_short)
                            self.SetFeatureTrue("%s arg1 %s arg2" % (left_context, between_context_short))
                            self.SetFeatureTrue("arg1 %s arg2 %s" % (between_context_short, right_context))

                        if a1[0] - a2[1] > 0:
                            left_context    = " ".join(tokenized[(a2[0]-k):a2[0]])
                            between_context = " ".join(tokenized[(a2[1]):a1[0]])
                            between_context_short = between_context
                            if a2[0] - a1[1] > 2 * k:
                                between_context_short = " ".join(tokenized[(a2[1]):a2[1]+k]) + "..." + " ".join(tokenized[(a1[0]-k):a1[0]])
                            right_context   = " ".join(tokenized[(a1[1]):(a1[1]+k)])


                            if a1[0] - a2[1] <= MAX_WORDS_BETWEEN:
                                self.SetFeatureTrue("arg2 %s arg1" % between_context)
                                self.SetFeatureTrue("%s arg2 %s arg1" % (left_context, between_context))
                                self.SetFeatureTrue("arg2 %s arg1 %s" % (between_context, right_context)) 

                            self.SetFeatureTrue("arg1 %s arg2" % between_context_short)
                            self.SetFeatureTrue("%s arg1 %s arg2" % (left_context, between_context_short))
                            self.SetFeatureTrue("arg1 %s arg2 %s" % (between_context_short, right_context))

                    ##########################################################
                    # POS between (example: u'arg1 VBN TO NNP arg2')
                    ##########################################################
#                    MAX_POS_BETWEEN = 8
#                    feature = None
#                    if a2[0] - a1[1] > 0 and a2[0] - a1[1] <= MAX_POS_BETWEEN:
#                        feature = "arg1 %s arg2" % " ".join(pos[(a1[1]):a2[0]])
#                    if a1[0] - a2[1] > 0 and a1[0] - a2[1] <= MAX_POS_BETWEEN:
#                        feature = "arg2 %s arg1" % " ".join(pos[(a2[1]):a1[0]])
#                    self.SetFeatureTrue(feature)

                    ##########################################################
                    # Nouns + verb between (example: u'arg2 ADV wed arg1')
                    ##########################################################
                    MAX_NV_BETWEEN = 8
                    feature = None
                    if a2[0] - a1[1] > 0 and a2[0] - a1[1] <= MAX_NV_BETWEEN:
                        left_context    = GetNV(tokenized[a1[0]-2:a1[0]], pos[a1[0]-2:a1[0]])
                        between_context = GetNV(tokenized[a1[1]:a2[0]], pos[a1[1]:a2[0]])
                        right_context   = GetNV(tokenized[a2[1]:a2[1]+2], pos[a2[1]:a2[1]+2])
                        self.SetFeatureTrue("arg1 %s arg2" % between_context)
                        self.SetFeatureTrue("%s arg1 %s arg2" % (left_context, between_context))
                        self.SetFeatureTrue("arg1 %s arg2 %s" % (between_context, right_context))
                    if a1[0] - a2[1] > 0 and a1[0] - a2[1] <= MAX_NV_BETWEEN:
                        left_context    = GetNV(tokenized[a2[0]-2:a2[0]], pos[a2[0]-2:a2[0]])
                        between_context = GetNV(tokenized[a2[1]:a1[0]], pos[a2[1]:a1[0]])
                        right_context   = GetNV(tokenized[a1[1]:a1[1]+2], pos[a1[1]:a1[1]+2])
                        self.SetFeatureTrue("arg2 %s arg1" % between_context)
                        self.SetFeatureTrue("%s arg2 %s arg1" % (left_context, between_context))
                        self.SetFeatureTrue("arg2 %s arg1 %s" % (between_context, right_context))

    def ComputeVerbFeatures(self):
        ##############################################
        # Features using T-NLP tag features
        ##############################################        
        #entityWords = self.arg1.lower().split(' ') + self.arg2.lower().split(' ')
        self.AddFeature('__BIAS__')
        for tweet in self.tweets:
            tokenized = tweet['words'].split(' ')
            pos       = tweet['pos'].split(' ')
            arg1      = tweet['arg1']
            arg2      = tweet['arg2']
            entityWords = arg1.lower().split(' ') + arg2.lower().split(' ')

            #sys.stderr.write("%s\t%s\n" % (str(tokenized), str(pos)))

#            for p in pos:
#                if p == 'VBD' or p == 'VBN':
#                    self.SetFeatureTrue('PAST_TENSE')

            verbs      = [tokenized[i].lower() for i in range(len(tokenized)) if (len(pos[i]) > 0 and (pos[i][0] == 'V'  or pos[i] == 'NN') and (not tokenized[i] in entityWords))]
            for v in verbs:
                self.SetFeatureTrue(v)

    #TODO
    def ComputeEventFeatures(self):
        ##############################################
        # Features using T-NLP tag features
        ##############################################        
        #entityWords = self.arg1.lower().split(' ') + self.arg2.lower().split(' ')
        self.AddFeature('__BIAS__')
        for tweet in self.tweets:
            tokenized = tweet['words'].split(' ')
            evenTags  = tweet['eventTags'].split(' ')
            arg1      = tweet['arg1']
            arg2      = tweet['arg2']
            entityWords = arg1.lower().split(' ') + arg2.lower().split(' ')

            events = GetSegments(tokenized, evenTags, 'EVENT')
            for e in events:
                self.SetFeatureTrue(e.lower())
    
    def AddFeature(self, feature):
        if isAscii(feature):
            self.features[feature] = self.features.get(feature, 0) + 1

    def SetFeatureTrue(self, feature):
        feature = feature.strip()
        if isAscii(feature):
            #sys.stderr.write("%s\n" % feature)
            self.features[feature] = 1
    
    def Features(self):
        return self.features

class FeatureSelection:
    def __init__(self, jsonData):
        self.jsonData = jsonData
        self.featureCounts = {}
        for d in self.jsonData:
            for f in d[0].keys():
                self.featureCounts[f] = self.featureCounts.get(f, 0.0) + 1.0
    
    def FilterFeaturesByCount(self, minCount):
        result = []
        for d in self.jsonData:
            newFeatures = {}
            for f in d[0].keys():
                if self.featureCounts.get(f) >= minCount:
                    newFeatures[f] = d[0][f]
            result.append([newFeatures, d[1]])
        return result

    def NfeaturesGeCount(self, minCount):
        n = 0
        for (f, count) in self.featureCounts.items():
            if count >= minCount:
                n += 1
        return n

class FeatureSelectionEntity:
    def __init__(self, tweets, featureData):
        self.features = featureData
        self.featureCounts = {}
        for i in range(len(self.features)):
            title = tweets[i]['title']
            arg2  = tweets[i]['arg2']
            ep = "%s\t%s" % (title, arg2)
            for f in self.features[i][0].keys():
                if not self.featureCounts.has_key(f):
                    self.featureCounts[f] = set()
                self.featureCounts[f].add(ep)
        for f in self.featureCounts.keys():
            self.featureCounts[f] = len(list(self.featureCounts[f]))
    
    def FilterFeaturesByCount(self, minCount):
        result = []
        for d in self.features:
            newFeatures = {}
            for f in d[0].keys():
                if self.featureCounts.get(f) >= minCount:
                    newFeatures[f] = d[0][f]
            result.append([newFeatures, d[1]])
        return result

    def NfeaturesGeCount(self, minCount):
        n = 0
        for (f, count) in self.featureCounts.items():
            if count >= minCount:
                n += 1
        return n
