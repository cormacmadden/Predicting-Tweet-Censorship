# You should download the archive and add the path of each year to paths list.

import os
import multiprocessing as mp
import json
import bz2

parallel = False # should the parsing run in parallel
processes = 20 # how many processes?
fileDir = os.path.dirname(__file__)
rawDataDir2018 = os.path.join(fileDir, '../Data/2018')

paths = [
    rawDataDir2018
    #add other year folders here if needed.
]

save_path = os.path.join(fileDir, '../Data/2018')
'U:\Machine_Learning_Module\Group Project\CensoredTweets\input'

def parse_withheld(file):
    date = '-'.join(file.split('/')[-5:-2])

    writer_withheld_tweets = open(save_path + date + "_withheldtweets.json", 'a')

    # print(file)
    bz_file = bz2.BZ2File(file)
    created_at = '-'.join(file.split('/')[-5:-2]) + " " + ':'.join(file.split('/')[-2:])[0:5] + ":00"

    # Check if the bz is broken
    try:
        bz_file_read = bz_file.read()
    except EOFError:
        return

    # get all lines in the bz file
    for line in bz_file_read.splitlines():

        # check if the json_obj in the line is broken
        try:
            json_obj = json.loads(line)
        except:
            continue

        if('created_at' in json_obj):
            created_at = json_obj['created_at']

            # check tweet
            if ('withheld_in_countries' in json_obj):
                json_obj['linked'] = 'no' # no retweet or quote
                writer_withheld_tweets.write(json.dumps(json_obj) + "\n")

            try:
                # handle retweet
                json_obj['retweeted_status']
                if('withheld_in_countries' in json_obj['retweeted_status']):
                    json_obj['linked'] = 'retweeted'
                    writer_withheld_tweets.write(json.dumps(json_obj) + "\n")
            except:
                # handle quote
                try:
                    json_obj['quoted_status']
                    if('withheld_in_countries' in json_obj['quoted_status']):
                        json_obj['linked'] = 'quoted'
                        writer_withheld_tweets.write(json.dumps(json_obj) + "\n")
                except:
                    continue


        else:
            try:
                # if the json object does not have a "created_at" then it should be a withheld message
                json_obj['withheld_in_countries']
                # store the approximate time of the withheld message, which is the creation time of the previous tweet
                json_obj['approx_time'] = created_at

            except:
                continue

    writer_withheld_tweets.close()

new_paths = []
for path in paths:
    months = os.listdir(path)
    for month in months:
        new_paths.append(path + "/" + month)

paths = new_paths
print(paths)

if(parallel): 
    pool = mp.Pool(processes)
    from datetime import datetime
else:
    pool = None
    from datetime import datetime

start = datetime.now()

for path in paths:
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith('.bz2'):
                files.append(os.path.join(r, file))

    print(path)
    print(len(files))

    if(parallel):
        pool.map(parse_withheld, [file for file in files])
    else:
        for file in files:
            parse_withheld(file)

if(parallel):
    pool.close()
    pool.join()

end = datetime.now()

print(start)
print(end)











