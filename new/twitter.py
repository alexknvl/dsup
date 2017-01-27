#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import string
from collections import namedtuple
import easytime


def compile_preprocessing_patterns():
    steps = [
        (r'http://[^\s]+', r' -URL- '),
        (r':-?\)', r' -SMI- '),
        (r':-?\(', r' -FRN- '),
        (r'\((\w)', ' ( \\1'),
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
        # Strips extra whitespace.
        (r'\s+', r' '),
        # Strips leading whitespace.
        (r'^\s+', r''),
        # Strips trailing whitespace.
        (r'\s+$', r''),
        # Normalize numbers (NOTE: this is new, just testing out.
        (r'[0-9]+', r' -NUM- ')
    ]

    return [(re.compile(step[0]), step[1]) for step in steps]

PREPROCESSING_STEPS = compile_preprocessing_patterns()

CONTROL_CHARS = ''.join(unichr(ch) for ch in range(0, 32) + range(127, 160))
CONTROL_CHAR_RE = re.compile('[%s]' % re.escape(CONTROL_CHARS))

ZH_COMMON = set([u'是', u'不', u'我', u'有', u'这', u'个', u'说', u'们',
                 u'为', u'你', u'时', u'那', u'去', u'过', u'对', u'她',
                 u'后', u'么'])
EN_COMMON = set(['it', 'he', 'she', 'going', 'day', 'tonight', 'with',
                 'just', 'want', 'make', 'the', 'you', 'about'])


def load_stopwords(path):
    with open(path) as input_file:
        for line in input_file:
            yield line.strip()

STOPWORDS = set(load_stopwords('../data/stop_words'))

# Load from file?
STOPLIST = set(['.', '"', "'", ',', "n't", "-", "(", ")", "#", "&", "!"])

TIMEREF_WORDS = set([
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
    'saturday', 'sunday',
    'mon', 'tues', 'wed', 'thur', 'thurs', 'fri', 'sat', 'sun',
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november',
    'december',
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
    'sep', 'sept', 'oct', 'nov', 'dec',
    '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th',
    '9th', '10th', '11th', '12th', '13th', '14th', '15th',
    '16th', '17th', '18th', '19th', '20th', '21st', '22nd',
    '23rd', '24th', '25th', '26th', '27th', '28th', '29th',
    '30th', '31st',
    'tomorrow'])


ChunkTag = namedtuple('ChunkTag', ['flag', 'type', 'weight'])


def parse_chunk_tag(tag_text):
    """
    >>> parse_chunk_tag('B-OTHER:1.19')
    ChunkTag(flag='B', type='OTHER', weight=1.19)

    >>> parse_chunk_tag('I-BLAH:1.19')
    ChunkTag(flag='I', type='BLAH', weight=1.19)

    >>> parse_chunk_tag('O')
    ChunkTag(flag='O', type=None, weight=0.0)
    """

    args = tag_text.split(':')
    flag = args[0][0]

    if len(args) > 1:
        ttype = args[0][2:]
        weight = float(args[1])
        return ChunkTag(flag, ttype, weight)
    else:
        return ChunkTag(flag, None, 0.0)


def format_chunk_tag(tag):
    """
    >>> format_chunk_tag(ChunkTag(flag='B', type='OTHER', weight=1.19))
    'B-OTHER:1.19'

    >>> format_chunk_tag(ChunkTag(flag='I', type='BLAH', weight=1.19))
    'I-BLAH:1.19'

    >>> format_chunk_tag(ChunkTag(flag='O', type=None, weight=0.0))
    'O'
    """

    if tag.type is not None:
        return '%s-%s:%s' % tag
    else:
        return tag.flag


def parse_chunk_tags(tags_text):
    """Parses a string consisting of chunk tags.

    >>> parse_chunk_tags('O B-ENTITY:1.19 I-ENTITY:1.19 O O B-ENTITY:1.00 O')
    ... # doctest: +NORMALIZE_WHITESPACE
    [ChunkTag(flag='O', type=None, weight=0.0),
     ChunkTag(flag='B', type='ENTITY', weight=1.19),
     ChunkTag(flag='I', type='ENTITY', weight=1.19),
     ChunkTag(flag='O', type=None, weight=0.0),
     ChunkTag(flag='O', type=None, weight=0.0),
     ChunkTag(flag='B', type='ENTITY', weight=1.0),
     ChunkTag(flag='O', type=None, weight=0.0)]

    >>> parse_chunk_tags('B-OTHER:1.19 I-OTHER:1.19')
    ... # doctest: +NORMALIZE_WHITESPACE
    [ChunkTag(flag='B', type='OTHER', weight=1.19),
     ChunkTag(flag='I', type='OTHER', weight=1.19)]
    """
    tags = tags_text.split(' ')
    for index, text in enumerate(tags):
        tags[index] = parse_chunk_tag(text)
    return tags


Tweet = namedtuple('Tweet',
                   ['sid', 'uid', 'loc',
                    'created_at', 'ref_date',
                    'entity', 'entity_type',
                    'words', 'pos_tags',
                    'ne_tags', 'event_tags'])


def parse_twitter_date(date_str):
    return easytime.strptime(date_str, '%a %b %d %H:%M:%S +0000 %Y', 'utc')


def format_twitter_date(timestamp):
    return easytime.strftime(timestamp, '%a %b %d %H:%M:%S +0000 %Y', 'utc')


def parse_tweet(line, created_at_is_formatted=False):
    fields = line.strip().split('\t')
    if len(fields) != 11:
        raise ValueError("Invalid tweet line.")

    sid, uid, loc, created_at, ref_date, entity, entity_type, \
        words, pos_tags, ne_tags, event_tags = fields

    sid = int(sid)
    uid = int(uid)

    if created_at_is_formatted:
        created_at = parse_twitter_date(created_at)
    else:
        created_at = float(created_at)

    words = words.split(' ')
    pos_tags = pos_tags.split(' ')
    ne_tags = parse_chunk_tags(ne_tags)
    event_tags = parse_chunk_tags(event_tags)

    assert len(words) == len(pos_tags)
    assert len(words) == len(ne_tags)
    assert len(words) == len(event_tags)

    return Tweet(sid, uid, loc,
                 created_at, ref_date,
                 entity, entity_type,
                 words, pos_tags,
                 ne_tags, event_tags)


def format_tweet(tweet, format_created_at=False):
    sid, uid, loc, created_at, ref_date, entity, entity_type, \
        words, pos_tags, ne_tags, event_tags = tweet

    sid = str(sid)
    uid = str(uid)

    if format_created_at:
        created_at = format_twitter_date(created_at)
    else:
        created_at = str(int(created_at))

    words = ' '.join(words)
    pos_tags = ' '.join(pos_tags)
    ne_tags = ' '.join(format_chunk_tag(x) for x in ne_tags)
    event_tags = ' '.join(format_chunk_tag(x) for x in event_tags)

    return '\t'.join((sid, uid, loc, created_at, ref_date, entity,
                      entity_type, words, pos_tags, ne_tags, event_tags))


def generate_chunk_segment_indices(chunk_tags):
    """
    >>> t = parse_chunk_tags('O B-ENTITY:1.2 I-ENTITY:1.2 O O B-ENTITY:1.0 O')
    >>> list(generate_chunk_segment_indices(t))
    [(1, 3), (5, 6)]

    >>> t = parse_chunk_tags('B-OTHER:1.2 I-OTHER:1.2 I-OTHER:1.2 B-OTHER:1.0')
    >>> list(generate_chunk_segment_indices(t))
    [(0, 3), (3, 4)]
    """
    start = -1

    for index, tag in enumerate(chunk_tags):
        if tag.flag == 'B':
            if start != -1:
                yield (start, index)
            start = index
        elif tag.flag == 'O' and start != -1:
            yield (start, index)
            start = -1

    if start != -1:
        yield (start, index + 1)


def generate_chunk_segments(words, chunk_tags, tag=None, lower=False):
    """
    >>> w = ['A', 'b', 'C', 'd', 'e', 'f', 'g']
    >>> t = parse_chunk_tags('O B-ENTITY:1.2 I-ENTITY:1.2 O O B-OTHER:1.0 O')
    >>> list(generate_chunk_segments(w, t))
    ['b C', 'f']

    >>> list(generate_chunk_segments(w, t, tag='ENTITY'))
    ['b C']

    >>> list(generate_chunk_segments(w, t, lower=True))
    ['b c', 'f']

    >>> words = ('RT @steverubel : Getting ' +
    ...          'started with Google multiple ' +
    ...          'account sign-in http://ow.ly/2lc83').split(' ')
    >>> event_tags = parse_chunk_tags(
    ...     'O O O O ' +
    ...     'B-EVENT:1.0002866712621914 O O O ' +
    ...     'O O O')
    >>> list(generate_chunk_segments(words, event_tags))
    ['started']
    """
    for start, end in generate_chunk_segment_indices(chunk_tags):
        if tag is None or chunk_tags[start].type == tag:
            segment = words[start:end]
            if lower:
                segment = (x.lower() for x in segment)
            yield ' '.join(segment)


def tweet_entities(tweet):
    return list(generate_chunk_segments(tweet.words, tweet.ne_tags,
                                        tag='ENTITY', lower=True))


def preprocess(text):
    for pattern, replacement in PREPROCESSING_STEPS:
        text = pattern.sub(replacement, text)
    return text


def strip_users(text):
    words = text.split(' ')
    return ' '.join(word for word in words if len(word) > 0 and word[0] != '@')


def strip_contractions(words):
    return [word for word in words if len(word) > 0 and word[0] != "'"]


def remove_control_chars(text):
    return CONTROL_CHAR_RE.sub('', text)


def is_ascii_string(text):
    try:
        text.decode('ascii')
    except UnicodeEncodeError:
        return False
    except UnicodeDecodeError:
        return False
    else:
        return True


def filter_stop(words):
    return [word for word in words if word.lower() not in STOPLIST]

if __name__ == "__main__":
    import doctest
    doctest.testmod()
