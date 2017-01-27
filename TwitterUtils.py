# -*- coding: utf-8 -*-

import re
import nltk
import string
import sys
import unicodedata

preprocessing = [
    (r'http://[^\s]+', r' -URL- '),
    (r':-?\)', r' -SMI- '),
    (r':-?\(', r' -FRN- '),
    (r'\((\w)', r' ( \1'),
    (r'\)(\w)', r' ) \1'),
    (r'\)\.', r' ) .'),
    (r'( |^)\$(\d+)( |$)', r' $ \2 '),
    (r'\.\.+', r' -DOTS- '),
    (r'@\w+', r' -USER- '),
    (r'( |^)[%s]+\w+( |$)' % string.punctuation, r' -EMO- '),
    (r'( |^)h[ha]+( |$)', r' -HA- '),
    (r'( |^)l[lo]+( |$)', r' -LOL- '),
    (r'(( |^)\w+)\.', r'\1 .'),
    #(r'#(\w+)', r' \1 '),
    (r'!+', r' -EXCLAM- '),
    (r'\?+', r' -QUESTION- '),
    (r'( |^)(1$|$1)[$1]*( |$)', r' -1D- '),
    (r'[%s]{2,}' % string.punctuation, r' -PSEQ- '),
    (r'\|+', r' -BAR- '),
    (r'\s+', r' '),   #Strip extra whitespace
    (r'^\s+', r''),   #Strip leading whitespace
    (r'\s+$', r''),   #Strip trailing whitespace
    (r'[0-9]+', r' -NUM- ')        #Normalize numbers (NOTE: this is new, just testing out
    ]

def stripUsers(string):
    words = string.split(' ')
    return ' '.join([word for word in words if(len(word) > 0 and word[0] != '@')])

def stripContractions(words):
    return [word for word in words if(len(word) > 0 and word[0] != "'")]

def preprocess(string):
    for p in preprocessing:
        string = re.sub(p[0], p[1], string)
    return string

control_chars = ''.join(map(unichr, range(0,32) + range(127,160)))
control_char_re = re.compile('[%s]' % re.escape(control_chars))

def removeControlChars(s):
    return control_char_re.sub('', s)

def isAscii(string):
    try:
        string.decode('ascii')
    except UnicodeEncodeError:
        return False
    except UnicodeDecodeError:
        return False
    else:
        return True

def containsKana(string):
    for c in string:
        if ord(c) >= 0x3040 and ord(c) <= 0x309f:
            return True
        elif ord(c) >= 0x30A0 and ord(c) <= 0x30FF:
            return True
    return False

def isAscii_old(string):
    for i in range(len(string)):
        o = ord(string[i])
        if o > 127:
            return False
    return True

ZH = set([u'是', u'不', u'我', u'有', u'这', u'个', u'说', u'们', u'为', u'你', u'时', u'那', u'去', u'过', u'对', u'她', u'后', u'么'])

EN = set(['it', 'he', 'she', 'going', 'day', 'tonight', 'with', 'just', 'want', 'make', 'the', 'you', 'about'])

stop_list = {}
#for line in open('stop_list'):
#    line = line.rstrip('\n')
#    stop_list[line] = 1
stop_list['.'] = 1
stop_list['"'] = 1
stop_list["'"] = 1
stop_list[','] = 1
stop_list["n't"] = 1
stop_list["-"] = 1
stop_list["("] = 1
stop_list[")"] = 1
stop_list["#"] = 1
stop_list["&"] = 1
stop_list["!"] = 1

def filterStop(words):
    return [word for word in words if(not stop_list.has_key(word.lower()))]

def isEnglish(string):
    if not isAscii(string):
        return False
    else:
        words = nltk.word_tokenize(string)
        for word in words:
            if word in EN:
                return True
        return False

def isChinese(string):
    for c in ZH:
        if c in string and not containsKana(string):
            return True
    return False

def IncHash2(hashtable, k1, k2, ammnt=1.0):
    if not hashtable.has_key(k1):
        hashtable[k1] = {}
    if not hashtable[k1].has_key(k2):
        hashtable[k1][k2] = 0.0
    hashtable[k1][k2] += ammnt

def IncHash(hashtable, k1, ammnt=1.0):
    if not hashtable.has_key(k1):
        hashtable[k1] = 0.0
    hashtable[k1] += ammnt

def GetHashCnt(hashtable, k1):
    if not hashtable.has_key(k1):
        return 0.0
    else:
        return hashtable[k1]

def WordDict2Wordle(wordDict, outFile):
    out = open(outFile, 'w')
    sortedWords = wordDict.keys()
    sortedWords.sort(lambda a,b: cmp(wordDict[b], wordDict[a]))
    for word in sortedWords:
        out.write("%s:%f\n" % (word, wordDict[word]))

