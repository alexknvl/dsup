import unittest
from collections import namedtuple
import fx

Segment = namedtuple('Segment', ['text', 'start', 'end'])

def get_segments(words, bio_tags):
    assert len(words) == len(bio_tags)

    results = []

    start = None
    for i in range(len(words)):
        if bio_tags[i][0] == 'B':
            if start != None:
                text = ' '.join(words[start:i])
                yield Segment(text=text, start=start, end=i)
            start = i
        elif bio_tags[i] == 'O' and start != None:
            text = ' '.join(words[start:i])
            yield Segment(text=text, start=start, end=i)
            start = None

    if start != None:
        text = ' '.join(words[start:])
        yield Segment(text=text, start=start, end=i)

# nell_categories = {}
# for line in open('../data/nell_categories_simple'):
#     fields = line.strip().split('\t')
#     if len(fields) != 2:
#         continue
#     (entity, category) = fields
#     entity = entity.strip()
#     category = category.strip()
#     if not nell_categories.has_key(entity):
#         nell_categories[entity] = []
#     nell_categories[entity].append(category)

# def GetCategory(entity):
#     result = []
#     if not nell_categories.has_key(entity):
#         return 'None'
#     lookup = entity
#     while nell_categories.has_key(lookup):
#         nc = nell_categories[lookup][0]         #TODO: handle more than one category.  This is probably fine for now...
#         if nc == 'agent' or nc == 'abstractthing' or nc == 'organization':
#             return lookup
#         lookup = nell_categories[lookup][0]
#     return 'None'

def replace_non_nv_words_with_tags(words, pos):
    assert len(words) == len(pos)
    return [w if len(pos[i]) > 0 and (pos[i][0] == 'N' or pos[i][0] == 'V')
            else pos[i]
            for i, w in enumerate(words)]


DEFAULT_MAX_WORDS_BETWEEN = 5
DEFAULT_MAX_POS_BETWEEN   = 8
DEFAULT_MAX_NV_BETWEEN    = 5
DEFAULT_CONTEXT_SIZES     = [1, 2, 3]
def generate_binary_features(args, words, pos, ne_tags,
                             max_words_between=DEFAULT_MAX_WORDS_BETWEEN,
                             max_pos_between=DEFAULT_MAX_POS_BETWEEN,
                             max_nv_between=DEFAULT_MAX_NV_BETWEEN,
                             context_sizes=DEFAULT_CONTEXT_SIZES):
    assert isinstance(args, tuple)
    assert len(args) == 2
    assert len(words) == len(pos) and len(pos) == len(ne_tags)

    words = [w.lower() for w in words]
    arg1, arg2 = args
    arg1 = arg1.lower()
    arg2 = arg2.lower()

    # A list of segments with matching name entities.
    segments = list(s for s in get_segments(words, ne_tags)
                    if s.text in args)

    for s1, s2 in fx.product(segments, segments):
        # If segments start on the same word, then they must be
        # the same segment (assuming get_segments works correctly).
        if s1.start == s2.start:
            continue

        if (s1.text, s2.text) == args:
            s1n, s2n = 'arg1', 'arg2'
        elif (s2.text, s1.text) == args:
            s2n, s1n = 'arg1', 'arg2'
        else:
            # If both named entites are the same, skip this segment pair.
            continue

        # If there is an intersection between segments or s1 is after s2,
        # skip this pair.
        if not (s1.end <= s2.start):
            continue

        # Words between (example: arg2 secretly wed arg1)
        for k in context_sizes:
            left    = words[s1.start-k:s1.start]
            between = words[s1.end:s2.start]
            right   = words[s2.end:s2.end+k]

            between_short = between
            if s1.end + 2 * k < s2.start:
                between_short =\
                    words[s1.end:s1.end+k] +\
                    ["..."] +\
                    words[s2.start-k:s2.start]

            if s1.end + max_words_between >= s2.start:
                yield ' '.join(left + [s1n] + between + [s2n])
                yield ' '.join(       [s1n] + between + [s2n])
                yield ' '.join(       [s1n] + between + [s2n] + right)

            yield ' '.join(left + [s1n] + between_short + [s2n])
            yield ' '.join(       [s1n] + between_short + [s2n])
            yield ' '.join(       [s1n] + between_short + [s2n] + right)

        # POS between (example: arg1 VBN TO NNP arg2)
        if s1.end + max_pos_between >= s2.start:
            yield ' '.join([s1n] + pos[s1.end:s2.start] + [s2n])

        # Nouns + verb between (example: arg2 ADV wed arg1)
        if s1.end + max_nv_between >= s2.start:
            left = replace_non_nv_words_with_tags(
                words[s1.start-2:s1.start],
                pos[  s1.start-2:s1.start])
            between = replace_non_nv_words_with_tags(
                words[s1.end:s2.start],
                pos[  s1.end:s2.start])
            right = replace_non_nv_words_with_tags(
                words[s2.end:s2.end+2],
                pos[  s2.end:s2.end+2])

            yield ' '.join(left + [s1n] + between + [s2n])
            yield ' '.join(       [s1n] + between + [s2n])
            yield ' '.join(       [s1n] + between + [s2n] + right)


# class FeatureExtractorBinary:
#     def __init__(self, tweets, target, arg1, arg2, keyword=None):
#         self.tweets = tweets
#         self.target = target
#         self.arg1 = arg1
#         self.arg2 = arg2
#         # for js in tweets:
#         #     #js.preprocessed = TwitterUtils.preprocess(' '.join(js.words))
#         #     js.preprocessed = TwitterUtils.preprocess(js.words)
#         self.features = {}
#         self.keyword = keyword
#
#     def ComputeEntityFeatures(self):
# #        for a1cat in nell_categories.get(self.arg1, []):
# #            self.SetFeatureTrue("arg1=%s" % a1cat)
#         self.SetFeatureTrue("arg1=%s" % GetCategory(self.arg1))
#
# #        for a2cat in nell_categories.get(self.arg2, []):
# #            self.SetFeatureTrue("arg2=%s" % a2cat)
#         self.SetFeatureTrue("arg2=%s" % GetCategory(self.arg2))
#
#     def ComputeTimexFeatures(self):
#         for tweet in self.tweets:
#             timeRef = None
#
#             try:
#                 timeRef = datetime.strptime(tweet.date, '%Y%m%d')
#             except:
#                 pass
#             try:
#                 timeRef = datetime.strptime(tweet.date, '%Y%m')
#             except:
#                 pass
#             try:
#                 timeRef = datetime.strptime(tweet.date, '%Y')
#             except:
#                 pass
#
#             if timeRef == None:
#                 continue
#
#             if timeRef > tweet.datetime + timedelta(days=1):
#                 self.SetFeatureTrue('FUTURE_TIME_REF')
#             if timeRef < tweet.datetime - timedelta(days=1):
#                 self.SetFeatureTrue('PAST_TIME_REF')
#             if timeRef < tweet.datetime - timedelta(days=365):
#                 self.SetFeatureTrue('VERY_OLD_TIME_REF')
#             if timeRef > tweet.datetime + timedelta(days=30):
#                 self.SetFeatureTrue('FAR_FUTURE_TIME_REF')
#
#     def ComputeVerbFeatures(self):
#         ##############################################
#         # Features using T-NLP tag features
#         ##############################################
#         #entityWords = self.arg1.lower().split(' ') + self.arg2.lower().split(' ')
#         for tweet in self.tweets:
#             tokenized = tweet.words.split(' ')
#             pos       = tweet.pos.split(' ')
#             arg1      = tweet.arg1
#             arg2      = tweet.arg2
#             entityWords = arg1.lower().split(' ') + arg2.lower().split(' ')
#
#             #sys.stderr.write("%s\t%s\n" % (str(tokenized), str(pos)))
#
# #            for p in pos:
# #                if p == 'VBD' or p == 'VBN':
# #                    self.SetFeatureTrue('PAST_TENSE')
#
#             verbs      = [tokenized[i].lower() for i in range(len(tokenized)) if (len(pos[i]) > 0 and (pos[i][0] == 'V'  or pos[i] == 'NN') and (not tokenized[i] in entityWords))]
#             for v in verbs:
#                 self.SetFeatureTrue(v)
#
#     #TODO
#     def ComputeEventFeatures(self):
#         ##############################################
#         # Features using T-NLP tag features
#         ##############################################
#         #entityWords = self.arg1.lower().split(' ') + self.arg2.lower().split(' ')
#         for tweet in self.tweets:
#             tokenized = tweet.words.split(' ')
#             evenTags  = tweet.eventTags.split(' ')
#             arg1      = tweet.arg1
#             arg2      = tweet.arg2
#             entityWords = arg1.lower().split(' ') + arg2.lower().split(' ')
#
#             events = GetSegments(tokenized, evenTags, 'EVENT')
#             for e in events:
#                 self.SetFeatureTrue(e.lower())

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.args = ("nick cannon", "mariah carey")
        self.words = ["NICK","CANNON","IS","MARRIED","TO","MARIAH","CAREY","?",
                      "WHERE","HAVE","I","BEEN","."]
        self.pos = ["CC","NNP","VBZ","VBN","TO","NNP","NNP",".",
                    "WRB","VBP","PRP","VBN","."]
        self.ne_tags = ["B","I","O","O","O","B","I","O","O","O","O","O","O"]

        self.assertEqual(len(self.words), len(self.pos))
        self.assertEqual(len(self.pos), len(self.ne_tags))

    def test_get_segments(self):
        a = list(get_segments(self.words, self.ne_tags))
        b = [Segment(text="NICK CANNON", start=0, end=2),
             Segment(text="MARIAH CAREY", start=5, end=7)]
        self.assertEqual(a, b)

    def test_replace_nv(self):
        a = replace_non_nv_words_with_tags(self.words, self.pos)

        self.assertEqual(len(a), len(self.words))
        for i in range(len(a)):
            if self.pos[i].startswith('V') or self.pos[i].startswith('N'):
                self.assertEqual(a[i], self.words[i])
            else:
                self.assertEqual(a[i], self.pos[i])

    def testGetBinaryFeatures(self):
        a = set(generate_binary_features(self.args, self.words,
                                         self.pos, self.ne_tags))
        b = set(['arg1 is married TO arg2',
                 'arg1 is ... to arg2 ?',
                 'arg1 is married TO arg2 . WRB',
                 'arg1 is married to arg2 ?',
                 'arg1 is ... to arg2',
                 'arg1 is married to arg2 ? where have',
                 'arg1 is married to arg2 ? where',
                 'arg1 VBZ VBN TO arg2',
                 'arg1 is married to arg2'])

        self.assertEqual(a, b)

        a = set(generate_binary_features(tuple(reversed(self.args)),
                                         self.words, self.pos, self.ne_tags))
        b = set(x.replace('arg1', '%temp%')
                 .replace('arg2', 'arg1')
                 .replace('%temp%', 'arg2')
                for x in b)

        self.assertEqual(a, b)
