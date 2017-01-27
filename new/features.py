#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import easytime
from twitter import \
    generate_chunk_segments, \
    generate_chunk_segment_indices, \
    parse_chunk_tags
from collections import Counter


def load_nell_categories_file(path):
    categories = {}

    with open(path) as input_file:
        for line in input_file:
            fields = line.strip().split('\t')
            if len(fields) != 2:
                continue
            (entity, category) = fields
            entity = entity.strip()
            category = category.strip()
            if entity not in categories:
                categories[entity] = []
            categories[entity].append(category)
    return categories

NELL_CATEGORIES = load_nell_categories_file('../data/nell_categories_simple')


def get_nell_category(entity):
    """
    >>> get_nell_category('christ')
    'humanagent'

    >>> get_nell_category('chess')
    'game'

    >>> get_nell_category('moscow')
    'geopoliticalentity'

    >>> get_nell_category('12e1dsad12e12e1') is None
    True
    """
    if entity not in NELL_CATEGORIES:
        return None
    lookup = entity
    while lookup in NELL_CATEGORIES:
        # TODO: Handle more than one category.
        category = NELL_CATEGORIES[lookup][0]
        if category == 'agent' or \
           category == 'abstractthing' or \
           category == 'organization':
            return lookup
        lookup = NELL_CATEGORIES[lookup][0]
    return None


def generate_nell_entity_features(arg1, arg2):
    """
    >>> list(generate_nell_entity_features('moscow', 'brad pitt'))
    ['arg1=geopoliticalentity', 'arg2=humanagent']
    """
    # for a1cat in nell_categories.get(self.arg1, []):
    #     self.SetFeatureTrue("arg1=%s" % a1cat)
    yield "arg1=%s" % get_nell_category(arg1)
    yield "arg2=%s" % get_nell_category(arg2)


def generate_timex_features(timestamp, timeref):
    """
    >>> import easytime
    >>> timestamp = easytime.ts(year=2010, month=8, day=1)
    >>> list(generate_timex_features(timestamp, '20100804'))
    ['FUTURE_TIME_REF']

    >>> list(generate_timex_features(timestamp, '200803'))
    ['PAST_TIME_REF', 'VERY_OLD_TIME_REF']

    >>> list(generate_timex_features(timestamp, '2012'))
    ['FUTURE_TIME_REF', 'FAR_FUTURE_TIME_REF']

    >>> list(generate_timex_features(timestamp, '20100801'))
    []
    """
    refts = None

    # FIXME: Doesn't take into account the timezone.
    #        Could feasibly parse the location & infer the TZ.
    parsers = [lambda t: easytime.strptime(t, '%Y%m%d', 'utc'),
               lambda t: easytime.strptime(t, '%Y%m', 'utc'),
               lambda t: easytime.strptime(t, '%Y', 'utc')]
    for parser in parsers:
        try:
            refts = parser(timeref)
            break
        except ValueError:
            pass

    if refts is not None:
        if refts > timestamp + easytime.dt(days=1):
            yield 'FUTURE_TIME_REF'
        if refts < timestamp - easytime.dt(days=1):
            yield 'PAST_TIME_REF'
        if refts < timestamp - easytime.dt(days=365):
            yield 'VERY_OLD_TIME_REF'
        if refts > timestamp + easytime.dt(days=30):
            yield 'FAR_FUTURE_TIME_REF'


def generate_nv_pos_features(arg1, arg2, words, pos_tags):
    """
    >>> words = ('Mr C says : " Say no to Him , her , ' +
    ...          'and your self ." Lol he kills me').split(' ')
    >>> pos_tags = ('NNP NNP VBZ : NNS VBP DT NN PRP , PRP , ' +
    ...             'CC PRP$ NN : UH PRP VBZ PRP').split(' ')
    >>> list(generate_nv_pos_features('Mr C', 'me', words, pos_tags))
    ['says', 'say', 'to', 'self', 'kills']
    """
    entity_words = arg1.lower().split(' ') + arg2.lower().split(' ')

    # if any(p == 'VBD' or p == 'VBN' for p in pos_tags):
    #     yield 'PAST_TENSE'

    for i, word in enumerate(words):
        pos_tag = pos_tags[i]
        if word in entity_words:
            continue

        if pos_tag[0] == 'V' or pos_tag == 'NN':
            yield word.lower()


def generate_event_features(words, event_tags):
    """
    >>> words = ('RT @steverubel : Getting ' +
    ...          'started with Google multiple ' +
    ...          'account sign-in http://ow.ly/2lc83').split(' ')
    >>> event_tags = parse_chunk_tags(
    ...          'O O O O ' +
    ...          'B-EVENT:1.0002866712621914 O O O ' +
    ...          'O O O')
    >>> list(generate_event_features(words, event_tags))
    ['started']
    """
    events = generate_chunk_segments(words, event_tags,
                                     tag='EVENT', lower=True)
    for e in events:
        yield e


def entity_indices(words, ne_tags, entities):
    result = [[] for e in entities]

    for start, end in generate_chunk_segment_indices(ne_tags):
        if ne_tags[start].type != 'ENTITY':
            continue

        segment = ' '.join(x.lower() for x in words[start:end])
        for i, entity in enumerate(entities):
            if segment == entity:
                result[i].append((start, end))
    return result


def generate_word_features_between(e1, e2, context_size, words,
                                   max_words_between):
    left_context = ' '.join(words[e1[1]-context_size:e1[1]])
    right_context = ' '.join(words[e2[2]:e2[2]+context_size])
    between_context = ' '.join(words[e1[2]:e2[1]])

    between_context_short = between_context
    if e2[1] - e1[2] > 2 * context_size:
        between_context_short = \
            ' '.join(words[e1[2]:e1[2]+context_size]) + \
            ' ... ' + \
            ' '.join(words[e2[1]-context_size:e2[1]])

    if e2[1] - e1[2] <= max_words_between:
        yield ' '.join((e1[0], between_context, e2[0]))
        yield ' '.join((left_context, e1[0], between_context, e2[0]))
        yield ' '.join((e1[0], between_context, e2[0], right_context))

    yield ' '.join((e1[0], between_context_short, e2[0]))
    yield ' '.join((left_context, e1[0], between_context_short, e2[0]))
    yield ' '.join((e1[0], between_context_short, e2[0], right_context))


def generate_pos_features_between(e1, e2, pos_tags, max_pos_between):
    between = ' '.join(pos_tags[e1[2]:e2[1]])
    if e2[1] - e1[2] <= max_pos_between:
        yield ' '.join((e1[0], between, e2[0]))


def merge_nv_pos(words, pos_tags):
    result = []
    for i, word in enumerate(words):
        pos_tag = pos_tags[i]
        if pos_tag[0] == 'N' or pos_tag[0] == 'V':
            result.append(word)
        else:
            result.append(pos_tags[i])
    return ' '.join(result)


def generate_nv_features_between(e1, e2, merged_tokens, max_nv_between):
    if e2[1] - e1[2] <= max_nv_between:
        left_context = merged_tokens[e1[1]-2:e1[1]]
        between_context = merged_tokens[e1[2]:e2[1]]
        right_context = merged_tokens[e2[2]:e2[2]+2]
        yield ' '.join((e1[0], between_context, e2[0]))
        yield ' '.join((left_context, e1[0], between_context, e2[0]))
        yield ' '.join((e1[0], between_context, e2[0], right_context))


def generate_lexical_features(arg1, arg2, words, ne_tags, pos_tags,
                              max_words_between=5,
                              max_pos_between=8,
                              max_nv_between=8):

    a1indices, a2indices = entity_indices(words, ne_tags, [arg1, arg2])
    merged_tokens = merge_nv_pos(words, pos_tags)

    for a1 in a1indices:
        for a2 in a2indices:
            if a1[0] == a2[0]:
                continue

            if a2[0] - a1[1] > 0:
                e1 = ('arg1', a1[0], a1[1])
                e2 = ('arg2', a2[0], a2[1])
            elif a1[0] - a2[1] > 0:
                e1 = ('arg2', a2[0], a2[1])
                e2 = ('arg1', a1[0], a1[1])
            else:
                continue

            # Words between (e.g. u'arg2 secretly wed arg1').
            for context_size in [1, 2, 3]:
                for f in generate_word_features_between(
                        e1, e2, context_size, words, max_words_between):
                    yield f

            # POS between (e.g. u'arg1 VBN TO NNP arg2').
            for f in generate_pos_features_between(
                    e1, e2, pos_tags, max_pos_between):
                yield f

            # Nouns + verb between (e.g. u'arg2 ADV wed arg1').
            for f in generate_nv_features_between(
                    e1, e2, merged_tokens, max_nv_between):
                yield f


def count_features(data):
    counter = Counter()
    for d in data:
        features = d[0]
        for f in features.iterkeys():
            counter[f] += 1
    return counter


def count_feature_entity_pairs(tweet_entities, tweet_features):
    counts = {}
    counter = Counter()

    for index, features in enumerate(tweet_features):
        arg1, arg2 = tweet_entities[index]

        ep = "%s\t%s" % (arg1, arg2)

        for f in features[0].iterkeys():
            if f not in counts:
                counts[f] = set()
            counts[f].add(ep)

    for f in counts.iterkeys():
        counter[f] = len(counts[f])

    return counter


def filter_features_by_count(data, min_count, feature_counts):
    for d in data:
        old_features = d[0]
        new_features = {}
        for f in old_features.iterkeys():
            if feature_counts[f] >= min_count:
                new_features[f] = old_features[f]

        result = list(d)
        result[0] = new_features
        yield result

if __name__ == "__main__":
    import doctest
    doctest.testmod()
