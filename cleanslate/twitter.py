import gzip
import ujson as json

import easytime

from common import Text, Sample, Edit, normalize_infobox


def parse_date_string(s):
    return easytime.strptime(s, '%a %b %d %H:%M:%S +0000 %Y', 'utc')


def parse_matched_line(line):
    (label, rel, arg1, arg2, tweet_date, edit_date,
     time_since_tweet, tweet_str, edit_str) = line.strip().split('\t')
    edit_json = json.loads(edit_str)
    tweet_json = json.loads(tweet_str)

    infobox = normalize_infobox(edit_json['infobox_name'])

    edit = Edit(id=edit_json['id'], timestamp=edit_json['timestamp'],
                relation=infobox + '/' + rel, args=(arg1, arg2))

    sid = tweet_json['sid']
    timestamp = parse_date_string(tweet_json['created_at'])

    words = tweet_json['words'].split(' ')
    pos = tweet_json['pos'].split(' ')
    ner = [s[0] for s in tweet_json['neTags'].split(' ')]
    text = Text(words=words, pos=pos, ner=ner)

    return Sample(id=sid, timestamp=timestamp, text=text,
                  args=(arg1, arg2), edits=[edit], y=-1, features=None)


def read_matched(path):
    with open(path) as input_file:
        for line in input_file:
            yield parse_matched_line(line)


def parse_unmatched_line(line):
    tweet_json = json.loads(line)

    sid = tweet_json['sid']
    arg1, arg2 = tweet_json['arg1'], tweet_json['arg2']
    timestamp = tweet_json['created_at']

    words = tweet_json['words'].split(' ')
    pos = tweet_json['pos'].split(' ')
    ner = tweet_json['neTags'].split(' ')
    text = Text(words=words, pos=pos, ner=ner)

    return Sample(id=sid, timestamp=timestamp, text=text,
                  args=(arg1, arg2), edits=[], y=-1, features=None)


def read_unmatched(path):
    with gzip.open(path) as input_file:
        for line in input_file:
            yield parse_unmatched_line(line)
