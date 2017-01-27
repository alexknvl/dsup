import json
import gzip
import urllib
import sys
from unidecode import unidecode
import redis
import re
import datetime
import os

from FeatureExtractor import GetSegments

REDIS = redis.Redis(host='localhost', port=6379, db=0)

api_key = open("%s/.freebase_api_key" % os.environ['DSUP_EVENT_DIR']).read()
service_url = 'https://www.googleapis.com/freebase/v1/mqlread'

def parseDate(dateStr):
    result = None
    try:
        result = datetime.datetime.strptime(dateStr, '%Y-%m-%d')
    except:
        pass
    try:
        result = datetime.datetime.strptime(dateStr, '%Y-%m-%dT%H:%M:%S')
    except:
        pass
    try:
        result = datetime.datetime.strptime(dateStr, '%Y-%m-%dT%H:%M:%SZ')
    except:
        pass
    return result


def normalizeString(string):
    if isinstance(string, unicode):
        return unidecode(string.strip().lower())
    else:
        return string.strip().lower()

def IndexMQL(mql_file):
    for fields in open(mql_file):
        (relation, query, extractor) = fields.split('\t')
        extractor = eval(extractor)
        params = {
                'query': query,
                'key': api_key
        }
        url = service_url + '?' + urllib.urlencode(params) + '&' + 'cursor'
        sys.stderr.write(url + "\n")
        response = json.loads(urllib.urlopen(url).read())
        #print "response:%s" % response
        while response.has_key('cursor') and response['cursor']:
            if not response.has_key('result'):
                print response
                break
            url = service_url + '?' + urllib.urlencode(params) + '&' + 'cursor=' + response['cursor']
            sys.stderr.write(url + "\n")
            for (arg1, arg2, date) in extractor(response['result']):
                if arg1 == None or arg2 == None or date == None:
                    continue
                if arg1 != None and arg2 != None and re.match('\d{4}-\d{2}-\d{2}', date):
                    (arg1, arg2) = (normalizeString(arg1), normalizeString(arg2))
                    print [relation, arg1, arg2, date]
                    sys.stdout.flush()
                    REDIS.sadd('fb::values::%s::%s' % (relation, arg1), arg2)
                    REDIS.sadd('fb::dates::%s::%s::%s' % (relation, arg1, arg2), date)
            response = json.loads(urllib.urlopen(url).read())
            if not response.has_key('cursor'):
                sys.stderr.write(str(response) + "\n")

def GetTrainingData(target_attribute):
    entityPairs = {}
    for ep in REDIS.smembers('%s:FBentityPairs' % target_attribute):
        (arg1, arg2) = ep.split("\t")
        #arg1 = ep
        #arg1 = '"%s" "%s"' % (arg1, arg2)
        for dateStr in REDIS.smembers('%s:FBentityPair:%s.dates' % (target_attribute, ep)):
            for tweet in [json.loads(x) for x in REDIS.smembers('%s:FBentityPair:%s.tweets' % (target_attribute, ep))]:
                #editDate  = datetime.datetime.fromtimestamp(edit['timestamp'])
                #editDate = datetime.datetime.strptime(dateStr, '%Y-%m-%d')
                editDate = parseDate(dateStr)
#                if editDate == None:
#                    continue
                #timestamp = (editDate - datetime.datetime(1970,1,1)) / datetime.timedelta(seconds=1)
                timestamp = (editDate.toordinal() - datetime.date(1970, 1, 1).toordinal()) * 24*60*60
                edit = {'title': arg1, 'timestamp': timestamp}
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

def IndexAlias(alias_file):
    F_IN = gzip.open(alias_file)
    counter = 0
    for line in F_IN:
        fields = line.split('\t')
        if fields[1] == '<http://rdf.freebase.com/ns/common.topic.alias>':
            if re.search(r'@en$', fields[2]) and len(fields[2]) > 2:
                alias = normalizeString(fields[2][1:len(fields[2])-4])
                mid = fields[0]
                REDIS.sadd('mid2alias:%s' % mid, alias)
                REDIS.sadd('alias2mid:%s' % alias, mid)
        counter += 1
        if counter % 1000000 == 0:
            sys.stderr.write("%s\n" % counter)

def GetAliases(entity):
    entity_aliases = [REDIS.smembers('mid2alias:%s' % x) for x in  REDIS.smembers('alias2mid:%s' % entity)]
    entity_aliases = [item for sublist in entity_aliases for item in sublist]   #flatten
    entity_aliases += [entity]
    return list(set([x.lower() for x in entity_aliases]))

def MatchFBAttribute(target_attribute):
    F_IN = gzip.open('/home/rittera/repos/backup/data/temporal_stream_ner_temp_event_new.gz')

    entityPairMatches = {}

    start_date = None
    end_date = None

    #Delete any old data
    REDIS.delete('%s:FBentityPairs' % target_attribute)

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

#        covered = set([])       #Don't create multiple matches for the same tweet
#        for entity_wp in GetAliases(entity):
        for entity_wp in set(list(REDIS.smembers('alias:%s' % entity)) + [entity]):
            attVals = list(REDIS.smembers('fb::values::%s::%s' % (target_attribute, entity_wp)))

            for av in attVals:
                #print (entity_wp, av)
                att_aliases = [x for x in REDIS.smembers('alias:%s' % av)]
                #att_aliases = GetAliases(av)

                for att_alias in set(list(att_aliases) + [av]):
#                    if (entity_wp, att_alias) in covered:
#                        continue
                    tweetEntities = [normalizeString(x) for x in GetSegments(words_list, neTags.split(' '), 'ENTITY')]
                    if att_alias in tweetEntities and not (att_alias in entity or entity in att_alias):
                        nmatches += 1
#                        covered.add((entity_wp, att_alias))

                        dates = REDIS.smembers('fb::dates::%s::%s::%s' % (target_attribute, entity_wp, av))
                        for date in dates:
                            editDate = None
                            try:
                                editDate = datetime.datetime.strptime(date, '%Y-%m-%d')
                            except:
                                pass
                            try:
                                editDate = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
                            except:
                                pass
                            try:
                                editDate = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')
                            except:
                                pass

                            if editDate == None:
                                continue

                            tweetDate = None
                            try:
                                tweetDate = datetime.datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
                            except Exception as e:
                                tweetDate = datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')

                            epKey = '%s\t%s' % (entity, att_alias)
                            if not entityPairMatches.has_key(epKey):
                                entityPairMatches[epKey] = {'dates':[], 'tweets':[]}

                            entityPairMatches[epKey]['dates'].append(date)
                            #entityPairMatches[epKey]['tweets'].append((entity, created_at, date, av, words, editDate, tweetDate, editDate - tweetDate))
                            entityPairMatches[epKey]['tweets'].append((entity, created_at, date, att_alias, words, editDate, tweetDate, editDate - tweetDate))

                            #Delete any old values
                            if not REDIS.sismember('%s:FBentityPairs' % target_attribute, epKey):
                                REDIS.delete('%s:FBentityPair:%s.dates' % (target_attribute, epKey))
                                REDIS.delete('%s:FBentityPair:%s.tweets' % (target_attribute, epKey))

                            REDIS.sadd('%s:FBentityPairs' % target_attribute, epKey)
                            REDIS.sadd('%s:FBentityPair:%s.dates'  % (target_attribute, epKey), date)
                            REDIS.sadd('%s:FBentityPair:%s.tweets' % (target_attribute, epKey), json.dumps({'sid':sid, 'uid':uid, 'loc':loc, 'created_at':created_at, 'date':date, 'entity':entity, 'eType':eType, 'words':words, 'pos':pos, 'neTags':neTags, 'eventTags':eventTags}))

                            print "------------------------------------------------------------"
                            #print "TWEET\t%s" % ("\t".join([unicode(x) for x in [entity, created_at, date, av, words, editDate, tweetDate]])).encode('utf-8')
                            try:
                                print "TWEET\t%s" % ("\t".join([str(x) for x in [entity, created_at, date, att_alias, words.encode('utf-8'), editDate, tweetDate]])).encode('utf-8')
                            except UnicodeDecodeError:
                                pass
                            print "TIMEDIFF\t%s" % (editDate - tweetDate)
                            print "EDITDATE\t%s" % editDate
                            print "------------------------------------------------------------"
                            sys.stdout.flush()
                            #if nmatches % 10 == 0:
                            if nmatches % 1 == 0:
                                sys.stdout.write('PROGRESS: %s\t%s\n' % (nlines, nmatches))

if __name__ == "__main__":
    eval(sys.argv[1])(sys.argv[2])
