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


def DumpFirstEdits(target_attributes):
    F_IN = gzip.open('/home/konovalo/data/all/wiki.gz')
    F_OUT = open('edits.log', 'w+')

    SPACE_RE = re.compile(r'\s+')

    # nLines = 0
    for line in F_IN:
        # nLines += 1
        # if nLines % 500 == 0:
        #     sys.stderr.write('%s\n' % nLines)

        js = json.loads(line.strip())

        for target_attribute in target_attributes:
            edits = InfoboxEdits(js, target_attribute)
            title = edits.title

            firstAttributeEdits = edits.GetFirstAttributeEdits()

            for attVal in firstAttributeEdits.keys():
                if len(firstAttributeEdits[attVal]['newvalue']) > 3 and len(firstAttributeEdits[attVal]['newvalue']) < 500:
                    F_OUT.write('%s\t%s\t%s\t%s\n' % (target_attribute, title, SPACE_RE.sub(' ', attVal).strip(), json.dumps(firstAttributeEdits[attVal]).encode('utf8', 'replace')))
    F_OUT.close()

if __name__ == "__main__":
    DumpFirstEdits(sys.argv[1].split(';'))
