import sys
import os
import urllib
import json
import re
from unidecode import unidecode

def normalizeString(string):
    if isinstance(string, unicode):
        return unidecode(string.strip().lower())
    else:
        return string.strip().lower()

#api_key = open("%s/.freebase_api_key" % os.environ['DSUP_EVENT_DIR']).read()
api_key = open("freebase_api_key").read()
service_url = 'https://www.googleapis.com/freebase/v1/mqlread'

def PrintMQL(mql_file):
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
            response = json.loads(urllib.urlopen(url).read())
            if not response.has_key('cursor'):
                sys.stderr.write(str(response) + "\n")

if __name__ == "__main__":
    PrintMQL(sys.argv[1])
