import re
import os
import sys
import gzip
import bz2
try:
    import ujson as json
except ImportError:
    import json
import redis
import datetime
import codecs
import unicodedata
import gigaword
import gigaword.utils
from unidecode import unidecode

from FeatureExtractor import GetSegments

import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'new'))
import easytime, fx

REDIS = redis.Redis(host='localhost', port=7778, db=0)


def normalizeString(string):
    if isinstance(string, unicode):
        return unidecode(string.strip().lower())
    else:
        return string.strip().lower()


def gigaword_files_in_range(attr, begin, end):
    # dir_path = '/data/anno_eng_gigaword_5/data/xml/'
    # for file_name in os.listdir(dir_path):
    # pattern = re.compile(r"Document ([A-Z_]+\d{6})\d{2}\.\d{4} : \d+")
    files = [l.rstrip() for l in open('indexGW_all_files.txt')]
    # last_file = [l for l in open('indexGW_' + attr + '.log')
    #              if pattern.match(l) is not None][-1]
    # name = pattern.match(last_file).groups()[0].lower()
    # index = [i for i, v in files if name in v][0]

    for i, file_name in enumerate(files):
        if i < 0.80 * len(files):
            continue

        date_str = file_name.split('_')[-1].split('.')[0]
        year = int(date_str[:4])
        month = int(date_str[4:6])
        good_date = (year == begin[0] and month >= begin[1]) or \
                    (year > begin[0] and year < end[0]) or \
                    (year == end[0] and month <= end[1])
        path = file_name#os.path.join(dir_path, file_name)
        if good_date:# and file_name.startswith('ltw'):
            yield path


SKIP_NE = set([
    'O', 'NUMBER', 'DATE', 'TIME', 'MONEY', 'PERCENT', 'MISC', 'ORDINAL'])


def expand_using_aliases(entity):
    aliases = list(REDIS.smembers('alias-1:%s' % entity))
    if len(aliases) > 0:
        entity_wp = aliases[0]
        return entity_wp, aliases
    else:
        return entity, []


def MatchAttribute(target_attribute, matchValue='newvalue'):
    #Delete any old data
    REDIS.delete('%s:entityPairs' % target_attribute)

    for file_path in gigaword_files_in_range(target_attribute, begin=(2004, 6), end=(2012, 3)):
        print "Reading %s" % file_path
        for doc in gigaword.read_file(file_path,
                                      parse_headline=False,
                                      parse_dateline=False,
                                      parse_coreferences=True,
                                      parse_sentences=True,
                                      parse_text=False):

            matches = 0
            for sentence in doc.sentences:
                entities = gigaword.utils.get_named_entities(
                    sentence,
                    skip_tags=SKIP_NE)
                entities = [e._replace(text=normalizeString(e.text))
                            for e in entities]

                matches += process_sentence(target_attribute, doc, sentence, entities)

            print "Document %s : %s" % (doc.id, matches)

def process_sentence(target_attribute, doc, sentence, entities):
    entity_set = set(e.text for e in entities)
    matches = 0
    for entity in entity_set:
        entity_wp, aliases = expand_using_aliases(entity)

        attVals = list(REDIS.smembers('fve::values::%s::%s' %
                                      (target_attribute, entity_wp)))

        firstInfoboxEditDate = None
        firstAttributeEditDate = None
        if REDIS.exists('fae:%s::%s' % (entity_wp, target_attribute)):
            firstInfoboxEditDate = datetime.datetime.fromtimestamp(
                int(list(REDIS.smembers('fie:%s' % entity_wp))[0]))
            firstAttributeEditDate = datetime.datetime.fromtimestamp(
                int(list(REDIS.smembers('fae:%s::%s' %
                    (entity_wp, target_attribute)))[0]))

        for av in attVals:
            att_aliases = [x for x in REDIS.smembers('alias:%s' % av)]

            for att_alias in set(att_aliases + [av]):
                if att_alias in entity_set and not (att_alias in entity or entity in att_alias):

                    matches += 1
                    edits = REDIS.smembers('fve::edits::%s::%s::%s' % (target_attribute, entity_wp, av))
                    for js in edits:
                        js = json.loads(js)

                        if len(js['newvalue']) > 5000:
                            continue

                        epKey = '%s\t%s' % (entity, att_alias)

                        #Delete any old values
                        if not REDIS.sismember('%s:entityPairs' % target_attribute, epKey):
                            REDIS.delete('%s:entityPair:%s.edits' % (target_attribute, epKey))
                            REDIS.delete('%s:entityPair:%s.tweets' % (target_attribute, epKey))

                        REDIS.sadd('%s:entityPairs' % target_attribute, epKey)
                        REDIS.sadd('%s:entityPair:%s.edits'  % (target_attribute, epKey), json.dumps(js))
                        REDIS.sadd(
                            '%s:entityPair:%s.tweets' % (target_attribute, epKey),
                            json.dumps({'sentence': sentence,
                                        'date': doc.date,
                                        'doc_id': doc.id}))
    return matches


def GetTrainingData(target_attribute):
    entityPairs = {}
    for ep in REDIS.smembers('%s:entityPairs' % target_attribute):
        (arg1, arg2) = ep.split("\t")

        for edit in [json.loads(x) for x in REDIS.smembers('%s:entityPair:%s.edits' % (target_attribute, ep))]:
            for tweet in [json.loads(x) for x in REDIS.smembers('%s:entityPair:%s.tweets' % (target_attribute, ep))]:
                print '\t'.join(str(x) for x in [
                    target_attribute, arg1, arg2,
                    json.dumps(edit), json.dumps(tweet)])


def sample_sentences(begin=(2004, 6), end=(2012, 3)):
    for file_path in gigaword_files_in_range(begin=(2004, 6), end=(2012, 3)):
        sys.stderr.write("Reading %s\n" % file_path)
        for doc in gigaword.read_file(file_path,
                                      parse_headline=False,
                                      parse_dateline=False,
                                      parse_coreferences=True,
                                      parse_sentences=True,
                                      parse_text=False):

            matches = 0
            for sentence in doc.sentences:
                entities = gigaword.utils.get_named_entities(
                    sentence,
                    skip_tags=SKIP_NE)
                entities = [e._replace(text=normalizeString(e.text))
                            for e in entities]

                for e1 in entities:
                    for e2 in entities:
                        if e1 != e2:
                            yield (e1.text, e2.text, e1.type, e2.type, {
                                'sentence': sentence,
                                'date': doc.date,
                                'doc_id': doc.id})


def ProcessAtt(string):
    return string.replace("::", " ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index Wikipedia infobox edits and match against Gigaword")
    parser.add_argument('--match_attribute', default=None)
    parser.add_argument('--oldvalue', action='store_true', default=False)
    parser.add_argument('--get_td',      default=None)
    parser.add_argument('--get_negative', action='store_true', default=False)
    args = parser.parse_args()

    if args.match_attribute:
        if args.oldvalue:
            MatchAttribute(ProcessAtt(args.match_attribute),
                           matchValue='oldvalue')
        else:
            MatchAttribute(ProcessAtt(args.match_attribute),
                           matchValue='newvalue')
            #print json.dumps(result, sort_keys=True, indent=4)

    if args.get_td:
        GetTrainingData(ProcessAtt(args.get_td))

    if args.get_negative:
        # begin=(2004, 6), end=(2012, 3)
        for e1, e2, e1t, e2t, s in fx.hash_sample(
                sample_sentences(begin=(2004, 6), end=(2011, 3)),
                key=lambda t: t[0] + '\t' + t[1],
                rate=100):
            print '\t'.join(str(x) for x in [0, e1, e2, e1t, e2t, json.dumps(s)])
        for e1, e2, e1t, e2t, s in fx.hash_sample(
                sample_sentences(begin=(2011, 4), end=(2012, 3)),
                key=lambda t: t[0] + '\t' + t[1],
                rate=1):
            print '\t'.join(str(x) for x in [1, e1, e2, e1t, e2t, json.dumps(s)])
