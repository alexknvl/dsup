import re
import sys
import gzip
import bz2
import json
import redis
import datetime
import codecs
import unicodedata
from unidecode import unidecode

from FeatureExtractor import GetSegments

import argparse

#from search import search

import cProfile, pstats, StringIO

REDIS = redis.Redis(host='localhost', port=7778, db=0)

def contains(small, big):
    for i in xrange(len(big)-len(small)+1):
        for j in xrange(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return i, i+len(small)
    return False

def normalizeString(string):
    if isinstance(string, unicode):
        return unidecode(string.strip().lower())
    else:
        return string.strip().lower()

##########################################################################
# Based on InfoboxEdits with the goal of extracting persistant edits
# for all attribute categories...
##########################################################################
class AllInfoboxEdits:
    def __init__(self, js):
        self.title = normalizeString(js['article_title'])
        self.edits = js['attribute']
        self.SortEdits()
        self.firstInfoboxEditTimestamp = None
        self.firstAttributeEditTimestamp = None
        if len(self.edits) > 0:
            self.firstInfoboxEditTimestamp = self.edits[0]['timestamp']
#        self.FilterEditsTarget(target_attribute)
        if len(self.edits) > 0:
            self.firstAttributeEditTimestamp = self.edits[0]['timestamp']

    def SortEdits(self):
        """ Sorts edits by timestamp """
        self.edits.sort(cmp=lambda a,b: cmp(a['timestamp'], b['timestamp']))

    def IsPersistentEdit(self, nv, j):
        if nv == '' or nv == None:
            return False

        editDate = datetime.datetime.fromtimestamp(self.edits[j]['timestamp'])
        attribute = self.edits[j]['key']

        for i in range(j+1,len(self.edits)):
            x1 = self.edits[i]
            if x1['key'] != attribute:
                continue

            timeToX1 = datetime.datetime.fromtimestamp(x1['timestamp']) - editDate

            if timeToX1 > datetime.timedelta(days=10):
                return True
            if (not x1.has_key('newvalue')) or (not nv in GetListValues(x1['newvalue'])):
                return False
        return True

    def GetFirstAttributeEdits(self):
        """ Find persistent attributes and the first edit to introduce them """
        coveredValues = set([])

        self.firstAttributeEdits = {}

        for i in range(len(self.edits)):
            edit = self.edits[i]
            attribute = edit['key']
            if not edit.has_key('newvalue') or edit['newvalue'] == '':
                continue
#            print "TITLE:%s" % self.title.encode('utf-8')
#            print "NEWVALUE:%s" % edit['newvalue'].encode('utf-8')
#            print "listValues:%s" % GetListValues(edit['newvalue'])
#            sys.stdout.flush()
            for attVal in GetListValues(edit['newvalue']):
#                if len(aliases) > 0:
#                    attVal = list(aliases)[0]
                if (attVal, attribute) in coveredValues:
                    continue
#                print "title-AV:(%s:::%s)" % (self.title, attVal)
#                print "ISPERSISTENT:%s" % self.IsPersistentEdit(attVal, i)
                if self.IsPersistentEdit(attVal, i):
                    coveredValues.add((attVal, attribute))
                    self.firstAttributeEdits[(attVal, attribute)] = edit
        return self.firstAttributeEdits

class InfoboxEdits:
    def __init__(self, js, target_attribute):
        self.title = normalizeString(js['article_title'])
        self.edits = js['attribute']
        self.SortEdits()
        self.firstInfoboxEditTimestamp = None
        self.firstAttributeEditTimestamp = None
        if len(self.edits) > 0:
            self.firstInfoboxEditTimestamp = self.edits[0]['timestamp']
        self.FilterEditsTarget(target_attribute)
        if len(self.edits) > 0:
            self.firstAttributeEditTimestamp = self.edits[0]['timestamp']

    def SortEdits(self):
        """ Sorts edits by timestamp """
        self.edits.sort(cmp=lambda a,b: cmp(a['timestamp'], b['timestamp']))

    def FilterEditsTarget(self, target_attribute):
        """ Remove all edits except the target """
        self.edits = [x for x in self.edits if x['key'].encode('utf-8') == target_attribute]

    def IsPersistentEdit(self, nv, j):
        if nv == '' or nv == None:
            return False
        editDate = datetime.datetime.fromtimestamp(self.edits[j]['timestamp'])
        for i in range(j+1,len(self.edits)):
            x1 = self.edits[i]
            timeToX1 = datetime.datetime.fromtimestamp(x1['timestamp']) - editDate
            if timeToX1 > datetime.timedelta(days=10):
                return True
            if (not x1.has_key('newvalue')) or (not nv in GetListValues(x1['newvalue'])):
                return False
        return True

    def GetFirstAttributeEdits(self):
        """ Find persistent attributes and the first edit to introduce them """
        coveredValues = set([])

        self.firstAttributeEdits = {}

        for i in range(len(self.edits)):
            edit = self.edits[i]
            if not edit.has_key('newvalue') or edit['newvalue'] == '':
                continue
            for attVal in GetListValues(edit['newvalue']):
                if attVal in coveredValues:
                    continue
                if self.IsPersistentEdit(attVal, i):
                    coveredValues.add(attVal)
                    self.firstAttributeEdits[attVal] = edit
        return self.firstAttributeEdits

def DumpPersistentAttributes():
    F_IN = gzip.open('/home/konovalo/data/all/wiki.gz')

    nLines = 0
    for line in F_IN:
        nLines += 1
        if nLines % 500 == 0:
            print nLines
            sys.stdout.flush()

        js = json.loads(line.strip())
        infoboxEdits = AllInfoboxEdits(js)
        edits = infoboxEdits.GetFirstAttributeEdits()

        #TODO: expand with aliases?
        for (attVal, attribute) in edits.keys():
            editDate = datetime.datetime.fromtimestamp(edits[(attVal, attribute)]['timestamp']).strftime("%Y%m%d")

#            for ta in tAliases:
#                for aa in aAliases:
#                    print "\t".join([normalizeString(infoboxEdits.title), ta, normalizeString(attribute), normalizeString(attVal), aa, editDate])
#            tAliases = list(REDIS.smembers('alias:%s' % infoboxEdits.title)) + [infoboxEdits.title]
#            aAliases = list(REDIS.smembers('alias:%s' % attVal)) + [attVal]

            #print "\t".join([normalizeString(infoboxEdits.title), attribute, normalizeString(attVal), editDate, json.dumps(edits[(attVal, attribute)])])
            print "\t".join([normalizeString(infoboxEdits.title), normalizeString(attribute), normalizeString(attVal), editDate])

def PrintAttribute(target_attribute):
    #F_IN = gzip.open('/nell/extdata/Google-WikiHistoricalInfobox/20120323-en-updates.json.gz')
    F_IN = gzip.open('/home/konovalo/data/all/wiki.gz')

    nLines = 0
    for line in F_IN:
        nLines += 1
        if nLines % 500 == 0:
            print nLines
            sys.stdout.flush()

        js = json.loads(line.strip())
        edits = InfoboxEdits(js, target_attribute)
        title = edits.title

        for edit in edits.edits:
            print (title, edit['key'], edit.get('oldvalue', None), edit.get('newvalue', None), datetime.datetime.fromtimestamp(edit['timestamp']), edit.get('comment',None))
            sys.stdout.flush()


def IndexAttribute(target_attribute):
    F_IN = gzip.open('/home/konovalo/data/all/wiki.gz')

    nLines = 0
    for line in F_IN:
        nLines += 1
        if nLines % 500 == 0:
            print nLines
            sys.stdout.flush()

        js = json.loads(line.strip())

        edits = InfoboxEdits(js, target_attribute)
        title = edits.title

        firstAttributeEdits = edits.GetFirstAttributeEdits()

        if edits.firstInfoboxEditTimestamp != None:
            REDIS.sadd('fie:%s' % title, edits.firstInfoboxEditTimestamp)
        if edits.firstAttributeEditTimestamp != None:
            REDIS.sadd('fae:%s::%s' % (title, target_attribute), edits.firstAttributeEditTimestamp)

        for attVal in firstAttributeEdits.keys():
            if len(firstAttributeEdits[attVal]['newvalue']) > 3 and len(firstAttributeEdits[attVal]['newvalue']) < 500:
                REDIS.sadd('fve::values::%s::%s' % (target_attribute, title), attVal)
                REDIS.sadd('fve::edits::%s::%s::%s' % (target_attribute, title, attVal), json.dumps(firstAttributeEdits[attVal]).encode('utf8', 'replace'))


def IndexAlias():
    #for line in bz2.BZ2File('/nell/extdata/freebase/freebase-datadump-quadruples.tsv.bz2', "r"):
    nlines = 0
    #for line in open('/home/rittera/repos/dsup_event/data/enwiki-redirect_converted.txt'):
    for line in codecs.open('/home/rittera/repos/dsup_event/data/enwiki-redirect_converted.txt', encoding='utf-8'):
        nlines += 1
        if nlines % 1000 == 0:
            print nlines
        fields = line.split('\t')
        if len(fields) != 2:
            continue
        (title, alias) = fields
        title = normalizeString(title)
        alias = normalizeString(alias)
        REDIS.sadd('alias:%s' % title, alias)
        REDIS.sadd('alias-1:%s' % alias, title)

def FirstAddEdit(k, allEdits):
    nv        = allEdits[k]['newvalue']
    timestamp = allEdits[k]['timestamp']

    if nv == '' or nv == None:
        return False

    editDate = datetime.datetime.fromtimestamp(timestamp)

    for i in range(k+1,len(allEdits)):
        x1 = allEdits[i]
        #if x1.has_key('newvalue') and x1['newvalue'] != '' and x1['newvalue'] == nv:
        if x1.has_key('newvalue') and nv in GetListValues(x1['newvalue']):
            firstEditDate = datetime.datetime.fromtimestamp(x1['timestamp'])

            if editDate != firstEditDate:
                return False

            for j in range(i,len(allEdits)):
                x2 = allEdits[j]
                timeToX2 = datetime.datetime.fromtimestamp(x2['timestamp']) - firstEditDate

                if not x2.has_key('newvalue'):
                    continue

                if not nv in GetListValues(x2['newvalue']):
                    return False
                elif timeToX2 > datetime.timedelta(days=30):
                    return True
    return True

            #How persistent is this edit?
#            timeToSecondEdit = None
#            if i < len(allEdits) - 1:
#                x2 = allEdits[i+1]
#                timeToSecondEdit = datetime.datetime.fromtimestamp(x2['timestamp']) - firstEditDate
#
#            if editDate == firstEditDate and ( timeToSecondEdit == None or timeToSecondEdit > datetime.timedelta(days=30) ):
#                return True
#            else:
#                return False


#TODO: Alias matching?

##############################################################################
# TODO: Handle WP links in old/new value.  Just extract the links....?
##############################################################################
#wikilink_rx = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]')
wikilink_rx = re.compile(r'[\[\{][\[\{](?:[^|\]]*\|)?([^\]]+)[\]\}][\]\}]')
def ExtractWPLinks(string):
    result = []
    for m in wikilink_rx.findall(string):
        result.append(m)
    return result

##############################################################################
# Wikipedia list valued attributes
##############################################################################
def GetListValues(string):
    result = []

    links = ExtractWPLinks(string)
    values = re.split(r'(\n|< *br */?>)', string)
    values = [re.sub('\(.*\)', '', x).strip() for x in values]
    values = [re.sub('<.*>', '', x).strip() for x in values]
    values = [re.sub('\{\{', '', x).strip() for x in values]
    values = [re.sub('\}\}', '', x).strip() for x in values]
    values = [re.sub('\[\[', '', x).strip() for x in values]
    values = [re.sub('\]\]', '', x).strip() for x in values]
    values = [re.sub('.*=',  '', x).strip() for x in values]

    result = values
    values = [re.split('[:,]', x)[0] for x in values]
    result += values

    result = result + links

    for i in range(len(result)):
        fields = re.split('\|', result[i])
        if len(fields) > 1:
            result[i] = fields[1]

    result = [normalizeString(x) for x in result]

    #result = [re.match(r'^(.*?)( \((.+)\))?$', x).groups()[0] for x in result]
    result = [x for x in result if len(x) > 2 and len(x) < 200 and x != u'none']

#    if len(string) < 1000:
#        print string.encode('utf-8')
#    print result
    return list(set(result))

def GetQueries(target_attribute):
    queryStartDates = {}
    queryEndDates   = {}
    queryCounts     = {}
    for ep in REDIS.smembers('%s:entityPairs' % target_attribute):
        (arg1, arg2) = ep.split("\t")
        #arg1 = ep
        #arg1 = '"%s" "%s"' % (arg1, arg2)
        for edit in [json.loads(x) for x in REDIS.smembers('%s:entityPair:%s.edits' % (target_attribute, ep))]:
            for tweet in [json.loads(x) for x in REDIS.smembers('%s:entityPair:%s.tweets' % (target_attribute, ep))]:
                editDate  = datetime.datetime.fromtimestamp(edit['timestamp'])
                tweetDate = None
                try:
                    tweetDate = datetime.datetime.strptime(tweet['created_at'], '%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    tweetDate = datetime.datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')

                #if ((editDate - tweetDate) >= datetime.timedelta(days=-1)) and ((editDate - tweetDate) < datetime.timedelta(days=300)):
                #if ((editDate - tweetDate) >= datetime.timedelta(days=-10)) and ((editDate - tweetDate) < datetime.timedelta(days=10)):
                #if ((editDate - tweetDate) >= datetime.timedelta(days=-10)) and ((editDate - tweetDate) < datetime.timedelta(days=100)):
                timeSinceTweet = editDate - tweetDate
                #if (timeSinceTweet >= datetime.timedelta(days=-2)) and (timeSinceTweet < datetime.timedelta(days=2)):
#                if (timeSinceTweet >= datetime.timedelta(days=-2)) and (timeSinceTweet < datetime.timedelta(days=100)):
#                if (timeSinceTweet >= datetime.timedelta(days=-2)) and (timeSinceTweet < datetime.timedelta(days=3)):
                if (timeSinceTweet >= datetime.timedelta(days=-10)) and (timeSinceTweet < datetime.timedelta(days=10)):
#                if (timeSinceTweet >= datetime.timedelta(days=-10)) and (timeSinceTweet < datetime.timedelta(days=100)):
                    key = '"%s" "%s"' % (arg1, arg2)
                    #print arg1
                    #print arg2
                    #print tweet['words']
                    print timeSinceTweet
                    print re.sub(arg2, "[[ %s ]]arg2" % arg2, re.sub(arg1, "[[ %s ]]arg1" % arg1, tweet['words'], flags=re.IGNORECASE), flags=re.IGNORECASE)
                    print "---------------------------------------------------------------------------------------------"
                    queryCounts[key] = queryCounts.get(key, 0) + 1
                    if not queryStartDates.has_key(key):
                        queryStartDates[key] = tweetDate
                        queryEndDates[key]   = tweetDate

                    if queryStartDates[key] > tweetDate:
                        queryStartDates[key] = tweetDate

                    if queryEndDates[key] < tweetDate:
                        queryEndDates[key] = tweetDate

                    #if (queryEndDates[key] - queryStartDates[key]) >= datetime.timedelta(days=30):
                    if (queryEndDates[key] - queryStartDates[key]) >= datetime.timedelta(days=5):
                        queryEndDates[key] = queryStartDates[key] + datetime.timedelta(days=5)

    for entity in sorted(queryStartDates.keys(), lambda a,b: cmp(queryCounts[b], queryCounts[a])):
        #print '%s\tmarried "%s" since:%s until:%s' % (queryCounts[entity], entity, queryStartDates[entity].strftime("%Y-%m-%d"), (queryEndDates[entity] + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
        #print '%s\twedding "%s" since:%s until:%s' % (queryCounts[entity], entity, queryStartDates[entity].strftime("%Y-%m-%d"), (queryEndDates[entity] + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
        #print '%s\twedding %s since:%s until:%s' % (queryCounts[entity], entity, queryStartDates[entity].strftime("%Y-%m-%d"), (queryEndDates[entity] + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
        print '%s\t%s since:%s until:%s' % (queryCounts[entity], entity, queryStartDates[entity].strftime("%Y-%m-%d"), (queryEndDates[entity] + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))

def GetTrainingData(target_attribute):
    entityPairs = {}
    for ep in REDIS.smembers('%s:entityPairs' % target_attribute):
        (arg1, arg2) = ep.split("\t")
        #arg1 = ep
        #arg1 = '"%s" "%s"' % (arg1, arg2)
        for edit in [json.loads(x) for x in REDIS.smembers('%s:entityPair:%s.edits' % (target_attribute, ep))]:
            for tweet in [json.loads(x) for x in REDIS.smembers('%s:entityPair:%s.tweets' % (target_attribute, ep))]:
                editDate  = datetime.datetime.fromtimestamp(edit['timestamp'])
                tweetDate = None
                try:
                    tweetDate = datetime.datetime.strptime(tweet['created_at'], '%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    tweetDate = datetime.datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')

                timeSinceTweet = editDate - tweetDate
                if (timeSinceTweet >= datetime.timedelta(days=-2)) and (timeSinceTweet < datetime.timedelta(days=3)):
                    print "+\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (target_attribute, arg1, arg2, tweetDate.strftime("%Y-%m-%d"), editDate.strftime("%Y-%m-%d"), timeSinceTweet.total_seconds(), json.dumps(tweet), json.dumps(edit))
                else:
                    print "-\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (target_attribute, arg1, arg2, tweetDate.strftime("%Y-%m-%d"), editDate.strftime("%Y-%m-%d"), timeSinceTweet.total_seconds(), json.dumps(tweet), json.dumps(edit))
#                    print timeSinceTweet
#                    print re.sub(arg2, "[[ %s ]]arg2" % arg2, re.sub(arg1, "[[ %s ]]arg1" % arg1, tweet['words'], flags=re.IGNORECASE), flags=re.IGNORECASE)
#                    print "---------------------------------------------------------------------------------------------"

def GetEvents(target_attribute):
    eventCounts = {}
    for ep in REDIS.smembers('%s:entityPairs' % target_attribute):
        for edit in [json.loads(x) for x in REDIS.smembers('%s:entityPair:%s.edits' % (target_attribute, ep))]:
            for tweet in [json.loads(x) for x in REDIS.smembers('%s:entityPair:%s.tweets' % (target_attribute, ep))]:
                editDate  = datetime.datetime.fromtimestamp(edit['timestamp'])
                tweetDate = None
                try:
                    tweetDate = datetime.datetime.strptime(tweet['created_at'], '%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    tweetDate = datetime.datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')

                if ((editDate - tweetDate) >= datetime.timedelta(days=-1)) and ((editDate - tweetDate) < datetime.timedelta(days=300)):
                    #print tweet['words']
                    #print tweet['eventTags']
                    #events = GetSegments(tweet['words'].split(' '), tweet['eventTags'].split(' '), 'EVENT')
                    words = tweet['words'].split(' ')
                    pos   = tweet['pos'].split(' ')
                    events = [normalizeString(words[i]) for i in range(len(words)) if pos[i][0] == 'V']
                    #events = GetSegments(tweet['words'].split(' '), tweet['eventTags'].split(' '), 'EVENT')
                    for event in events:
                        eventCounts[event] = eventCounts.get(event, 0) + 1

    for (e,c) in sorted(eventCounts.items(), lambda a,b: cmp(a[1], b[1])):
        print "%s\t%s" % (e,c)


editCache = {}
#A bit of caching for efficiency...
def GetEdits(entity, target_attribute):
    if entity == 'tripoli':
        #This one is too annoying...
        return None
    if not editCache.has_key('title:%s:%s' % (entity, target_attribute)):
        if REDIS.exists('title:%s:%s' % (entity, target_attribute)):
            edits = REDIS.smembers('title:%s:%s' % (entity, target_attribute))
            edits = [json.loads(x) for x in edits]
            edits = sorted(edits, cmp=lambda a,b: cmp(a['timestamp'], b['timestamp']))
            editCache['title:%s:%s' % (entity, target_attribute)] = edits
    return editCache.get('title:%s:%s' % (entity, target_attribute), None)

#def GetNegativeData(sampleRate=1000):
#def GetNegativeData(sampleRate=500):
def GetNegativeData(sampleRate=1):
    #F_IN = gzip.open('/home/rittera/repos/backup/data/temporal_stream_ner_temp_event.gz')
    #F_IN = gzip.open('/home/rittera/repos/backup/data/temporal_stream_ner_temp_event_new.gz')
    F_IN = gzip.open('/home/rittera/data/temporal_stream_ner_temp_event_new.gz')


    lastsid = None

    nLines = 0
    for line in F_IN:
        nLines += 1
        if nLines % sampleRate != 0:
            continue

        fields = line.strip().split('\t')
        if len(fields) != 11:
            continue
        (sid, uid, loc, created_at, date, entity, eType, words, pos, neTags, eventTags) = fields

        if lastsid and lastsid == sid:
            continue
        lastsid = sid

        tweetDate = None
        try:
            tweetDate = datetime.datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            tweetDate = datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')

        words_list = words.lower().split(' ')

        entities = GetSegments(words_list, neTags.split(' '), 'ENTITY')

        for i in range(len(entities)):
            for j in range(len(entities)):
                arg1 = entities[i]
                arg2 = entities[j]
                if i != j and arg1 != arg2:
                    print "%s\t%s\t%s\t%s" % (arg1, arg2, tweetDate.strftime("%Y-%m-%d"), json.dumps({'sid':sid, 'uid':uid, 'loc':loc, 'created_at':created_at, 'date':date, 'entity':entity, 'eType':eType, 'words':words, 'pos':pos, 'neTags':neTags, 'eventTags':eventTags}))
                    sys.stdout.flush()



def MatchAttribute(target_attribute, matchValue='newvalue'):
    #F_IN = codecs.getreader('utf-8')(gzip.open('/home/rittera/repos/backup/data/temporal_stream_ner_temp_event.gz'))
    #F_IN = gzip.open('/home/rittera/repos/backup/data/temporal_stream_ner_temp_event.gz')
    F_IN = gzip.open('/home/rittera/repos/backup/data/temporal_stream_ner_temp_event_new.gz')

    entityPairMatches = {}

    start_date = None
    end_date = None

    #Delete any old data
    REDIS.delete('%s:entityPairs' % target_attribute)

    nlines = 0
    nmatches = 0
    for line in F_IN:
        nlines += 1

        fields = line.strip().split('\t')
        if len(fields) != 11:
            continue
        (sid, uid, loc, created_at, date, entity, eType, words, pos, neTags, eventTags) = fields
        entity = normalizeString(entity)
        words_list = words.lower().split(' ')

        entity_wp = entity
        aliases = list(REDIS.smembers('alias-1:%s' % entity))
        if len(aliases) > 0:
            entity_wp = aliases[0]

        attVals = list(REDIS.smembers('fve::values::%s::%s' % (target_attribute, entity_wp)))

        firstInfoboxEditDate   = None
        firstAttributeEditDate = None
        if REDIS.exists('fae:%s::%s' % (entity_wp, target_attribute)):
            firstInfoboxEditDate   = datetime.datetime.fromtimestamp(int(list(REDIS.smembers('fie:%s' % entity_wp))[0]))
            firstAttributeEditDate = datetime.datetime.fromtimestamp(int(list(REDIS.smembers('fae:%s::%s' % (entity_wp, target_attribute)))[0]))

        for av in attVals:
            att_aliases = [x for x in REDIS.smembers('alias:%s' % av)]

            for att_alias in set(list(att_aliases) + [av]):
                tweetEntities = [normalizeString(x) for x in GetSegments(words_list, neTags.split(' '), 'ENTITY')]
                if att_alias in tweetEntities and not (att_alias in entity or entity in att_alias):
                    nmatches += 1

                    edits = REDIS.smembers('fve::edits::%s::%s::%s' % (target_attribute, entity_wp, av))
                    for js in edits:
                        js = json.loads(js)

                        if len(js['newvalue']) > 5000:
                            continue

                        editDate  = datetime.datetime.fromtimestamp(js['timestamp'])
                        tweetDate = None
                        try:
                            tweetDate = datetime.datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
                        except Exception as e:
                            tweetDate = datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')

                        epKey = '%s\t%s' % (entity, att_alias)
                        if not entityPairMatches.has_key(epKey):
                            entityPairMatches[epKey] = {'edits':[], 'tweets':[]}

                        entityPairMatches[epKey]['edits'].append(json.dumps(js))
                        #entityPairMatches[epKey]['tweets'].append((entity, created_at, date, av, words, editDate, tweetDate, editDate - tweetDate))
                        entityPairMatches[epKey]['tweets'].append((entity, created_at, date, att_alias, words, editDate, tweetDate, editDate - tweetDate))

                        #Delete any old values
                        if not REDIS.sismember('%s:entityPairs' % target_attribute, epKey):
                            REDIS.delete('%s:entityPair:%s.edits' % (target_attribute, epKey))
                            REDIS.delete('%s:entityPair:%s.tweets' % (target_attribute, epKey))

                        REDIS.sadd('%s:entityPairs' % target_attribute, epKey)
                        REDIS.sadd('%s:entityPair:%s.edits'  % (target_attribute, epKey), json.dumps(js))
                        REDIS.sadd('%s:entityPair:%s.tweets' % (target_attribute, epKey), json.dumps({'sid':sid, 'uid':uid, 'loc':loc, 'created_at':created_at, 'date':date, 'entity':entity, 'eType':eType, 'words':words, 'pos':pos, 'neTags':neTags, 'eventTags':eventTags}))

                        print "------------------------------------------------------------"
                        #print "TWEET\t%s" % ("\t".join([unicode(x) for x in [entity, created_at, date, av, words, editDate, tweetDate]])).encode('utf-8')
                        try:
                            print "TWEET\t%s" % ("\t".join([str(x) for x in [entity, created_at, date, att_alias, words.encode('utf-8'), editDate, tweetDate]])).encode('utf-8')
                        except UnicodeDecodeError:
                            pass
                        print "TIMEDIFF\t%s" % (editDate - tweetDate)
                        print "TIMESINCEFIRSTEDIT\t%s" % (editDate - firstInfoboxEditDate)
                        print "FIRSTEDITDATE\t%s" % firstInfoboxEditDate
                        print "EDITJS\t%s" % js
                        print "EDITDATE\t%s" % editDate
                        print "------------------------------------------------------------"
                        sys.stdout.flush()
                        #if nmatches % 10 == 0:
                        if nmatches % 1 == 0:
                            sys.stdout.write('PROGRESS: %s\t%s\n' % (nlines, nmatches))

PROFILE=False

def ProcessAtt(string):
    return string.replace("::", " ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index Wikipedia infobox edits and match against Twitter Events")
    parser.add_argument('--index_attribute', default=None)
    parser.add_argument('--print_attribute', default=None)
    parser.add_argument('--match_attribute', default=None)
    parser.add_argument('--get_events',      default=None)
    parser.add_argument('--get_td',      default=None)
    parser.add_argument('--get_queries',     default=None)
    parser.add_argument('--index_alias', action='store_true', default=False)
    parser.add_argument('--get_negative', action='store_true', default=False)
    parser.add_argument('--oldvalue',    action='store_true', default=False)
    parser.add_argument('--dump_persistent_edits',    action='store_true', default=False)
    args = parser.parse_args()

    if PROFILE:
        pr = cProfile.Profile()
        pr.enable()

    if args.index_attribute:
        IndexAttribute(ProcessAtt(args.index_attribute))

    if args.print_attribute:
        PrintAttribute(ProcessAtt(args.print_attribute))

    if args.get_td:
        GetTrainingData(ProcessAtt(args.get_td))

    if args.match_attribute:
        if args.oldvalue:
            MatchAttribute(ProcessAtt(args.match_attribute), matchValue='oldvalue')
        else:
            MatchAttribute(ProcessAtt(args.match_attribute), matchValue='newvalue')
            #print json.dumps(result, sort_keys=True, indent=4)

    if args.get_events:
        GetEvents(ProcessAtt(args.get_events))

    if args.index_alias:
        IndexAlias()

    if args.get_queries:
        GetQueries(ProcessAtt(args.get_queries))

    if args.get_negative:
        GetNegativeData()

    if args.dump_persistent_edits:
        DumpPersistentAttributes()

    if PROFILE:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
