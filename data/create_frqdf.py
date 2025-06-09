import json
from pathlib import Path
import numpy as np
import pandas as pd
import os
from collections import Counter
from all_spraak import systematic_sampler
import datetime
import argparse

merger = {
    "N1C_berikareX": "N1_berikare",
    "N1C_förortsgängX": "N1_förortsgäng",
    "N1C_globalistX": "N1_globalist",
    "N1C_kulturberikarX": "N1_kulturberikare",
    "N1C_återvandringsX": "N1_återvandring",
    "N2C_återvandrarX": "N2_återvandrare",
}

def get_corpus_freq(path_to_json, spec=None, intervall="quarterly", y0=2000):
    corpus_freq = dict()
    with open(Path(path_to_json), encoding="utf-8") as f:
        data = json.loads(f.read())
    for corpus in data["corpora"].keys():
        
        if spec != None:
            # print("XXX")
            if corpus.lower() != spec.lower():
                continue
        print(corpus)
        for time, freq in data["corpora"][corpus].items(): 
            time = time[:4] + "-" + time[4:]
            time = datetime.datetime.strptime(time, "%Y-%m")
            t = systematic_sampler(time, intervall, y0)
            # t = str(t)
            if t in corpus_freq:
                corpus_freq[t] += freq
            else:
                corpus_freq[t] = freq

    return corpus_freq

def get_tf(path, corpus_freq):
    TF  = dict()
    FPM = dict()
    path = Path(path)
    for time_file in os.listdir(path):
        counter = Counter()
        with open(path / time_file, encoding="utf-8") as f:
            for line in f.readlines():
                term, _ = tuple(line.split("\t"))
                if term in merger:
                    counter.update([merger[term]])
                else:
                    counter.update([term])
        time = time_file.replace(".txt", "")
        time = int(time)
        TF[time] = counter
        FPM[time] = {t: (f / corpus_freq[time]) for t, f in counter.items()} # there is no adjustment for millions (?!)
    return TF, FPM

def main(args):

    CORP_FREQ = get_corpus_freq(
        path_to_json = args.corp_freq_json, 
        spec         = args.spec_corp,
        intervall    = args.intervall,   # "quarterly" 
        y0           = args.year_zero    # 2000
    )

    tf, fpm = get_tf(
        path        = args.path_to_context_vectors, 
        corpus_freq = CORP_FREQ
    )
    
    terms = set()
    for k in tf.keys():
        for q in tf[k].keys():
            terms.add(q)

    assert tf.keys() == fpm.keys()

    times = tf.keys()
    times = list(range(min(times), max(times)+1))
    #print(times)

    data = []

    for term in sorted(terms):
        d = {"Term": term}
        for time in times:
            if time in tf:
                if term in tf[time]:
                    d[f"frq_{time}"] = tf[time][term]
                else:
                    d[f"frq_{time}"] = np.nan
            else:
                d[f"frq_{time}"] = np.nan

        for time in times:
            if time in fpm:
                if term in fpm[time]:
                    d[f"fpm_{time}"] = fpm[time][term]
                else:
                    d[f"fpm_{time}"] = np.nan
            else:
                d[f"fpm_{time}"] = np.nan
        data.append(d)

    df = pd.DataFrame(data)
    df.set_index("Term", inplace=True)
    df.to_csv(args.df_path, sep="\t")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="create_frqdf.py", description="Creates dataframe with frequenices over time from `context` dir and total corpus counts (JSON).")
    parser.add_argument("corp_freq_json", type=str, help="Path to total corpus counts (JSON).") # "../../data/Corpora/corpus_freq/flashback-all.json"
    parser.add_argument("path_to_context_vectors", type=str, help="Path to `context` dir from which files the term frequencies are *counted*.") # "../../data/Corpora/flashback-plt-time/quarterly/contexts/vectors/bert-base-swedish-cased/"
    parser.add_argument("df_path", type=str, help="Path to output dataframe") # "../../data/Corpora/term_freq/freq_fb_plt_quarterly.csv"
    parser.add_argument("--spec_corp", type=str, help="Specify a corpus in `corp_freq_json`. Default: None.") # "flashback-politik"
    parser.add_argument("--year_zero", type=int, default=2000, help="Provide to define the first year (y0). Default: 2000.") 
    parser.add_argument("--intervall", type=str, choices=["yearly", "semiannually", "quarterly", "monthly"], default="monthly", help="Provide to define intervall. Default = monthly.")

    args = parser.parse_args()

    for k, v in args.__dict__.items():
        print(k, ":::", v)

    main(args)

    print()
    print("Done!")