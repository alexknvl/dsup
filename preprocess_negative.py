import sys, errno, os, os.path
import ujson as json
import gzip
import easytime
from unidecode import unidecode

def parse_date(string):
    return easytime.strptime(string, '%a %b %d %H:%M:%S +0000 %Y', 'utc')
def inside_range(range, x):
    return range[0] <= x and x <= range[1]

# TRAIN_RANGE = (easytime.ts(year=2008, month=9, day=1),
#                easytime.ts(year=2011, month=6, day=1))
# DEV_RANGE   = (easytime.ts(year=2011, month=6, day=5),
#                easytime.ts(year=2012, month=1, day=1))

TRAIN_RANGE = (easytime.ts(year=2004, month=6, day=1),
               easytime.ts(year=2009, month=6, day=1))
DEV_RANGE   = (easytime.ts(year=2009, month=6, day=5),
               easytime.ts(year=2012, month=4, day=1))

KNOWN_SIDS_PATH = '/home/konovalo/dsup_event/new-data/annotated_ids.csv'
KNOWN_SIDS = set(line.strip() for line in open(KNOWN_SIDS_PATH) if line.strip() != '')


def parse_unmatched_line(line):
    try:
        _, arg1, arg2, ne1, ne2, sentence_str = \
            line.decode('utf-8').strip().split('\t')
    except:
        return None

    sentence_json = json.loads(sentence_str)

    doc_id = sentence_json['doc_id']
    sentence_id = sentence_json['sentence'][0]
    sid = "%s.%s" % (doc_id, sentence_id)
    timestamp = easytime.ts(*sentence_json['date'])

    arg1 = unidecode(arg1).lower()
    arg2 = unidecode(arg2).lower()

    return (arg1, arg2, sid, timestamp, line)


if __name__ == "__main__":
    file_name = sys.argv[1]
    sampling_rate = int(sys.argv[2])

    total_read = 0
    total_printed = 0
    total_known = 0
    total_broken = 0

    # idEps = set([])
    for line in gzip.open(sys.argv[1]):
        total_read += 1
        if total_read % 100000 == 0:
            sys.stderr.write(
                "%s/%s/%s/%s (%s)\n" %
                (total_broken, total_known, total_printed, total_read,
                 float(total_printed) / total_read))

        parsed_line = parse_unmatched_line(line)
        if parsed_line is None:
            total_broken += 1
            continue

        arg1, arg2, sid, timestamp, line = parsed_line
        isInTrainRange = inside_range(TRAIN_RANGE, timestamp)
        isInDevRange   = inside_range(DEV_RANGE, timestamp)
        isKnown = sid in KNOWN_SIDS

        # (arg1, arg2, tweetDate, tweetStr) = line.strip().split('\t')
        #
        # tweet = json.loads(tweetStr)
        # tweet['created_at'] = int(parse_date(tweet['created_at']))
        #
        # tweet['arg1'] = arg1
        # tweet['arg2'] = arg2
        #
        # del tweet['entity']
        # del tweet['eType']
        # #del tweet['date']
        # del tweet['loc']
        # del tweet['uid']
        # del tweet['eventTags']
        # if 'from_date' in tweet:
        #     del tweet['from_date']
        # tweet['neTags'] = ' '.join(t[0] for t in tweet['neTags'].split(' '))
        #
        # isInTrainRange = inside_range(TRAIN_RANGE, tweet['created_at'])
        # isInDevRange   = inside_range(DEV_RANGE, tweet['created_at'])
        # isKnown = str(tweet['sid']) in KNOWN_SIDS

        if not (isInTrainRange or isInDevRange):
            continue

        #sys.stderr.write(str(KNOWN_ARGS) + '\n')

        # if hash("%s\t%s" % (arg1, arg2)) % sampling_rate != 0:
        #     continue

        if isInTrainRange:
            if hash("%s\t%s" % (arg1,arg2)) % sampling_rate != 0:
                continue
        else:
            if hash("%s\t%s" % (arg1,arg2)) % sampling_rate != 0:
                continue
            if isKnown:
                total_known += 1
            else:
                # continue
                pass

        # idEp = tweet['sid'] + '\t' + arg1 + '\t' + arg2
        # if idEp in idEps:
        #     continue
        # idEps.add(idEp)

        # sys.stdout.write(json.dumps(tweet) + '\n')
        print line
        total_printed += 1
