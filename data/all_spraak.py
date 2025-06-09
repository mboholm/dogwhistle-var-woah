import requests
import re
import json
from pathlib import Path
import time
import datetime
import os
import argparse
import shutil

def read_json(path):
    with open(Path(path), encoding="utf-8") as f:
        JSON = json.loads(f.read())
    return JSON

def get_config(path, verbose = True):
    JSON = read_json(path)

    keywords = []
    POSs = []
    for entry in JSON["keywords"]:
        keywords.append(entry["keyword"])
        POSs.append(entry["pos"])

    kws = [(keyword, pos) for keyword, pos in zip(keywords, POSs)]

    if JSON["corpus_prefix_only"]:
        corpora = []
        for corpus in JSON["corpus"]:
            corpora.extend(sb_corpora(corpus))
    else:
        corpora = JSON["corpus"]

    if verbose:
        print("Keywords:", kws)
        print("Corpus:", corpora)
    
    return kws, corpora  

def sb_corpora(start_with_criterion = None):
    response = requests.get("https://ws.spraakbanken.gu.se/ws/korp/v8/info")
    response = response.json()
    corpora  = [corpus.lower() for corpus in response["corpora"]]
    
    if start_with_criterion != None:
        corpora = [corpus for corpus in corpora if corpus.startswith(start_with_criterion)]
    
    return corpora 

def corpus_keyword_param(corpus, keyword, pos, mode):
    param = [f'corpus={corpus}']
    if mode == "lexeme":
        assert pos != None, "For mode == 'lexeme', provide POS!"
        param.append(f'cqp=[lex contains "{keyword}\\.\\.{pos}\\.1"]')
    elif mode == "form":
        param.append(f'cqp=[word="{keyword}.*"]')
    else:
        print("func `corpus_keyword_param` needs a `mode` = {'lexeme', 'form'}")

    return param 

def param2str(corpus, keyword, pos, default_context, end, mode="lexeme"):
    base = corpus_keyword_param(corpus=corpus, keyword=keyword, pos=pos, mode=mode)
    param = [
        f'end={end}',
        f'default_context={default_context}',
        f'show=pos,lemma,ref,dephead,deprel',
        f'show_struct=text_date'
    ]
    param = "&".join(base + param)

    return param

def count(corpus, keyword, pos, mode="lexeme", temporal=False):

    func = "count_time" if temporal else "count"

    param = corpus_keyword_param(corpus=corpus, keyword=keyword, pos=pos, mode=mode)

    api_url = f'https://ws.spraakbanken.gu.se/ws/korp/v8/{func}?{"&".join(param)}'
    response = requests.get(api_url)
    payload = response.json()

    return payload

def one_search(corpus, keyword, pos, mode, default_context="1+sentence", end=10, redefine_end = False):
    param = param2str(corpus=corpus, keyword=keyword, pos=pos, mode=mode, default_context=default_context, end=end)
    
    api_url = f'https://ws.spraakbanken.gu.se/ws/korp/v8/query?{param}' # query_sample
    # 'https://ws.spraakbanken.gu.se/ws/korp/v8/query?corpus=flashback-politik&end=1000&default_context=1+sentence&cqp=[word="kulturberika.*"]&show=msd,lemma&show_struct=text_date'

    response = requests.get(api_url)
    payload = response.json()
    if "ERROR" in payload.keys():
        print(payload["ERROR"])

    if redefine_end:
        new_end  = payload["hits"]
        param    = param2str(corpus=corpus, keyword=keyword, pos=pos, mode=mode, default_context=default_context, end=new_end)
        new_url  = f'https://ws.spraakbanken.gu.se/ws/korp/v8/query?{param}'
        response = requests.get(new_url)
        payload  = response.json()
        if "ERROR" in payload.keys():
            print(payload["ERROR"])
        
    return payload

def saveMyJson(payload, path):
    path = Path(path)
    with open(path, mode = "w", encoding = "utf-8") as f:
        json.dump(payload, f, indent=4)

def multiple_counts(kws, corpora, dir_path, mode, temporal=False):
    t0 = time.time()
    dir_path = Path(dir_path)
    for corpus in corpora:
        for kw, pos in kws:
            t = time.time() - t0
            print(f"{corpus:<35}{kw:<10}{int(t/60):>3}:{int(t%60):<3}", end="\r")
            payload = count(corpus=corpus, keyword=kw, pos=pos, mode=mode, temporal=temporal)
            saveMyJson(payload, dir_path / f"{corpus}_{kw}_{pos}.json") 

def multiple_queries(kws, corpora, dir_path, mode, end = 10, redefine_end = False):
    t0 = time.time()
    dir_path = Path(dir_path)
    for corpus in corpora:
        for kw, pos in kws:
            t = time.time() - t0
            print(f"{corpus:<35}{kw:<10}{int(t/60):>3}:{int(t%60):<3}", end="\r")            
            payload = one_search(corpus=corpus, keyword=kw, pos=pos, mode=mode, default_context="1+sentence", end = end, redefine_end = redefine_end)
            saveMyJson(payload, dir_path / f"{corpus}_{kw}_{pos}.json")  

# def find_first(path):
#     years = []
#     for json in os.listdir(path):
#         for ex in read_json(path / json)["kwic"]:
#             t = ex["structs"]["text_date"].split()[0]
#             year = int(t.split("-")[0])
#             # year = datetime.datetime.strptime(t, "%Y-%m-%d").year
#             years.append(year)

#     return min(years)

def systematic_sampler(time, intervall, y0):
    # match case
    match intervall:
        case "yearly":
            return time.year

        case "semiannually":
            x = 0 if time.month < 7 else 1
            return ((time.year - y0) * 2) + x

        case "quarterly": 
            if time.month < 4:
                x = 0
            elif time.month > 3 and time.month < 7:
                x = 1
            elif time.month > 6 and time.month < 10:
                x = 2
            else:
                x = 3
            return ((time.year - y0) * 4) + x

        case "monthly":
            return ((time.year - y0) * 12) + time.month # Should be `time.month - 1`
            

def tmp2simple(path, out_dir, intervall, y0):
    path = Path(path)
    #y0 = find_first(path)
    d = dict()
    for json in os.listdir(path):
        json = read_json(path / json)
        for ex in json["kwic"]:
            time = ex["structs"]["text_date"].split()[0]
            time = datetime.datetime.strptime(time, "%Y-%m-%d")
            t = systematic_sampler(time, intervall, y0)
            #month = ((time.year - y0) * 12) + time.month 
            sentence = " ".join([token["word"] for token in ex["tokens"]])
            if t in d:
                d[t].append(sentence)
            else: 
                d[t] = [sentence]

    for t in d.keys():
        with open(Path(out_dir) / f"{t}.txt", encoding = "utf-8", mode = "w") as f:
            for sentence in d[t]:
                f.write(sentence + "\n")

def main(args):
    kws, corpora = get_config(args.config)

    if kws[0][-1] == None: # i.e., no POS provided for keywords
        search_mode = "form"
    else:
        search_mode = "lexeme"
    
    if args.mode == "count":
        multiple_counts(kws, corpora, dir_path = args.output, mode = search_mode, temporal = args.temporal)

    if args.mode == "collect" and args.simple_format == True:
        # create tmp in args.output
        tmp = Path(args.output) / "tmp"
        os.mkdir(tmp) 
    
        multiple_queries(kws, corpora, dir_path = tmp, mode=search_mode, redefine_end = True)

        # tmp2simple
        tmp2simple(tmp, args.output, args.intervall, args.genesis)

        if not args.keep_tmp:
            #os.rmdir(tmp)
            shutil.rmtree(tmp, ignore_errors=True)
        
    if args.mode == "collect" and args.simple_format == False:

        multiple_queries(kws, corpora, dir_path = args.output, mode=search_mode, redefine_end = True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="all-spraak", description="Get data from Språkbanken.")
    parser.add_argument("mode", type=str, choices=["collect", "count"], help="Mode: collect sentences with keywords (`collect`) or get counts of keywords (`count`).")
    parser.add_argument("output", type=str, help="Path to output directory.")
    parser.add_argument("config", type=str, help="Path to JSON config file.")
    parser.add_argument("-g", "--genesis", type=int, default=2000, help="Provide for `mode=collect` with `--simple_format`, to define the first year (y0). Default: 2000.")
    parser.add_argument("-t", "--temporal", action='store_true', help="Provide for `mode=count` if get temporal counts.")
    parser.add_argument("-s", "--simple_format", action='store_true', help="Provide for `mode=collect` to save sentences in separate files named by its month from the first month in the collected data. Creates and uses output_dir/tmp for payload from Språkbanken.")
    parser.add_argument("-n", "--intervall", type=str, choices=["yearly", "semiannually", "quarterly", "monthly"], default="monthly", help="Provide for `mode=collect` with `--simple_format` to define intervall. Default = monthly.")
    parser.add_argument("-k", "--keep_tmp", action='store_true', help="Provide for `mode=collect` with `--simple_format` to keep output_dir/tmp (payload from Språkbanken).")

    args = parser.parse_args()

    print(args.__dict__)

    main(args)

    print()
    print("Done!")

# args = Arguments()
# args.mode = "collect"
# args.config = "c:/Users/xbohma/Desktop/scratch/gripes2/config/test_spraak_data.json"
# args.output = "c:/Users/xbohma/Desktop/scratch/data/corpus/fm"
# args.temporal = True
# args.simple_format = True
# args.keep_tmp = True # default false
# main(args)        