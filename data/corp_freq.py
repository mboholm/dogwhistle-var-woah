import requests
#import re
import json
from pathlib import Path
import time
import datetime
import os
import argparse
#import shutil

def saveMyJson(payload, path):
    path = Path(path)
    with open(path, mode = "w", encoding = "utf-8") as f:
        json.dump(payload, f, indent=4)

def sb_corpora(start_with_criterion = None):
    response = requests.get("https://ws.spraakbanken.gu.se/ws/korp/v8/info")
    response = response.json()
    corpora  = [corpus.lower() for corpus in response["corpora"]]
    
    if start_with_criterion != None:
        corpora = [corpus for corpus in corpora if corpus.startswith(start_with_criterion)]
    
    return corpora 

def corpus_counts(corpus, granularity):
    if type(corpus) == list:
        corpus = ",".join(corpus) # https://spraakbanken.gu.se/korp/#?cqp=%5B%5D&corpus=flashback-resor,flashback-sport&search=lemgram%7Cinnebandy%5C.%5C.nn%5C.1

    api_url = f"https://ws.spraakbanken.gu.se/ws/korp/v8/timespan?corpus={corpus}&granularity={granularity}" # ...corpus=flashback-resor&granularity=m
    response = requests.get(api_url)
    payload = response.json()

    return payload

def main(args):
    t0 = time.time()

    if args.prefix:
        corpus = sb_corpora(args.corpus)
    else:
        corpus = args.corpus

    payload = corpus_counts(corpus, args.granularity)

    saveMyJson(payload, args.output_path)

    t = time.time() - t0

    print("Done!", int(t), "s.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="corp_freq.py", description="Get corpus frequencies from Språkbanken.")
    parser.add_argument("corpus", type=str, help="Corpus to get frequencies for.")
    parser.add_argument("output_path", type=str, help="Path to directory for output.")
    parser.add_argument("-p", "--prefix", action="store_true", help="Use `corpus` as prefix and collect frequencies for all corpora starting with `corpus`.")
    parser.add_argument("-r", "--granularity", type=str, default="y", help="Granularity of counts. 'm' for month. Default: 'y' (year). See: https://ws.spraakbanken.gu.se/docs/korp#tag/Statistics/paths/~1timespan/")

    args = parser.parse_args()

    print(args.__dict__)

    main(args)

    print()
    print("Done!")