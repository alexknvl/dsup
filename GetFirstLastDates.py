import gzip
import sys
import json
from datetime import *


def InfoboxFirstLastDates():
    minTimestamp = None
    maxTimestamp = None
    for line in gzip.open('/nell/extdata/Google-WikiHistoricalInfobox/20120323-en-updates.json.gz'):
        js = json.loads(line.strip())

        for edit in js['attribute']:
            ts = long(edit['timestamp'])
            if minTimestamp == None or ts < minTimestamp:
                minTimestamp = ts
            if maxTimestamp == None or ts > maxTimestamp:        
                maxTimestamp = ts
                print (datetime.fromtimestamp(minTimestamp), datetime.fromtimestamp(maxTimestamp))
                sys.stdout.flush()

def TwitterFirstLastDates():
    minDate = None
    maxDate = None
    for line in gzip.open('/home/rittera/repos/backup/data/temporal_stream_ner_temp_event.gz'):
        fields = line.strip().split('\t')
        if len(fields) != 11:
            continue
        (sid, uid, loc, created_at, date, entity, eType, words, pos, neTags, eventTags) = fields

        tweetDate = None
        try:
            tweetDate = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            tweetDate = datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')

        
        if minDate == None or tweetDate < minDate:
            minDate = tweetDate
            print (minDate, maxDate)
            sys.stdout.flush()
        if maxDate == None or tweetDate > maxDate:
            maxDate = tweetDate

if __name__ == "__main__":
    TwitterFirstLastDates()

