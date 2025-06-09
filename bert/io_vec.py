import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from difflib import SequenceMatcher
from pathlib import Path
from sklearn.utils.extmath import softmax
from collections import Counter
import logging
import time
import datetime
import os
import json
import argparse


def keyness(trg, ref, min_frq = 3, verbose = True): # Consider metric
    
    d = dict()
    
    trg_tot = len(trg)
    ref_tot = len(ref)
    
    for w in trg.keys():
        if trg[w] < min_frq:
            continue
        if w in ref:
            d[w] = (trg[w] / trg_tot) / (ref[w] / ref_tot) # Odds Ratio (OR)
        else:
            d[w] = np.inf
    
    if verbose:
        for word, trg_freq, keyness  in sorted([(w, trg[w], k) for w, k in d.items()], key = lambda x: x[1], reverse = True)[:20]:
            if word in ref:
                ref_freq = ref[word]
            else:
                ref_freq = 0
            print(f"{word:<20}{trg_freq:<4}{(trg_freq/trg_tot):<6.3f}{ref_freq:<4}{(ref_freq/ref_tot):<6.3f}{keyness:.4}")        
    
    return d

def inspect(
    df,               # Replacement Dataframe
    dwe,              # Dog Whistle Expression
    meaning,          # 1 for ingroup, 2 for outgroup
    phase,            # 1 for first phase of data collection, 2 for second phase
    sw = None,        # stopwords
    punct = None,     # remove punctuations
    verbose = True,
    multi = False,    # Keep the multi-word units of the replacements
    rel_freq = False, # use relative frequncies freq / no. of documents
    lower_input = True
):
    
    counter = Counter()
    
    if type(df) == pd.DataFrame:
        column = df.loc[df[f"{dwe}_w{phase}_C"] == meaning, f"{dwe}_text_w{phase}"]
    else:
        column = df
    
    for x in column:
        if lower_input:
            x = x.lower()        

        if punct != None:
            for p in punct:
                x = x.replace(p, "")
        x = x.split()
        if sw != None:
            x = [w for w in x if w not in sw]
        
        if multi:
            x = ["_".join(x)]
        
        counter.update(set(x)) # Obs. terms are only counted once per "document"
    
    if rel_freq:
        counter = Counter({w: c/len(column) for w,c in counter.items()})
        
    if verbose:
        for w, f in sorted(counter.items(), key = lambda x: x[1], reverse = True)[:15]:
            print(f"{w:<30}{f}")
        print("-----------------------")
        print("Total no. of types:", len(counter))

    return counter

def select_A(
    df,             # Replacement Dataframe
    dwe,            # Dog Whistle Expression
    phase = "both", # 1 for first phase of data collection, 2 for second phase, "both" for both
    sw = None,      # stopwords
    punct = None,   # remove punctuations
    k = None,
    min_freq = None,
    min_OR = None,
    empty_intersect = False
):
    
    if type(k) == tuple:
        k_in, k_out = k
    else:
        k_in  = k
        k_out = k
    if type(min_freq) == tuple:
        min_freq_in, min_freq_out = min_freq
    else:
        min_freq_in  = min_freq
        min_freq_out = min_freq
    if type(min_OR) == tuple:
        min_OR_in, min_OR_out = min_OR
    else:
        min_OR_in  = min_OR
        min_OR_out = min_OR
    
    if phase == "both":
        x = pd.concat([
            df.loc[df[f"{dwe}_w{1}_C"] == 1, f"{dwe}_text_w{1}"],
            df.loc[df[f"{dwe}_w{2}_C"] == 1, f"{dwe}_text_w{2}"]
        ]).to_list()
                
        y = pd.concat([
            df.loc[df[f"{dwe}_w{1}_C"] == 2, f"{dwe}_text_w{1}"],
            df.loc[df[f"{dwe}_w{2}_C"] == 2, f"{dwe}_text_w{2}"]
        ]).to_list()

        ingroup = inspect(x, dwe, None, None, sw, punct, verbose = False, rel_freq = True)
        outgroup = inspect(y, dwe, None, None, sw, punct, verbose = False, rel_freq = True)

        keyness_in2out = keyness(ingroup, outgroup, verbose = False, min_frq = -1)
        keyness_out2in = keyness(outgroup, ingroup, verbose = False, min_frq = -1)
        
    else:    
    
        ingroup = inspect(df, dwe, 1, phase, sw, punct, verbose = False, rel_freq = True)
        outgroup = inspect(df, dwe, 2, phase, sw, punct, verbose = False, rel_freq = True)
        keyness_in2out = keyness(ingroup, outgroup, verbose = False, min_frq = -1)
        keyness_out2in = keyness(outgroup, ingroup, verbose = False, min_frq = -1)
    
    A_in  = [w for w in ingroup.keys()]
    A_out = [w for w in outgroup.keys()]
    
    if empty_intersect:
        A_in  = [w for w in A_in if w not in outgroup.keys()]
        A_out = [w for w in A_out if w not in ingroup.keys()]
        
    if min_freq != None:
        A_in  = [w for w in A_in if ingroup[w] >= min_freq_in]
        A_out = [w for w in A_out if outgroup[w] >= min_freq_out]
    
    if min_OR != None:
        A_in  = [w for w in A_in if keyness_in2out[w] >= min_OR_in]
        A_out = [w for w in A_out if keyness_out2in[w] >= min_OR_out] # too strict to have the same threshold for both
        
    if k != None:
        A_in  = [w for w,_ in sorted(ingroup.items(), key = lambda x: x[1], reverse = True) if w in A_in][:k_in]
        A_out = [w for w,_ in sorted(outgroup.items(), key = lambda x: x[1], reverse = True) if w in A_out][:k_out]
    
    
    return A_in, A_out

def matcher(string, A_list, punct): 
    
    match = False

    for p in punct:
        string = string.replace(p, "")
        
    string = string.replace("/", " ")
    
    for w in string.split(" "):
        if w.lower() in A_list: # NB. lower(); replacaments.txt are taken unmodifed from xlsx-file 
            match = True
            
    return match

def load_replacements(dwe, meaning, rnd, model, data_path):

    if rnd == None:
        with open(data_path / dwe / meaning / "replacements.txt", encoding="utf-8") as f:
            idx_t, text = zip(*[tuple(line.strip("\n").split("\t")) for line in f.readlines()]) 
        with open(data_path / dwe / meaning / "vectors" / model / "vecs.txt", encoding="utf-8") as f:
            lines = [tuple(line.strip("\n").split("\t")) for line in f.readlines()]
            lines = [(idx, [float(v) for v in vec.split()]) for idx, vec in lines]
            idx_v, vectors = zip(*lines)
            
    else:
        with open(data_path / dwe / meaning / rnd / "replacements.txt", encoding="utf-8") as f:
            idx_t, text = zip(*[tuple(line.strip("\n").split("\t")) for line in f.readlines()]) 
        with open(data_path / dwe / meaning / rnd / "vectors" / model / "vecs.txt", encoding="utf-8") as f:
            lines = [tuple(line.strip("\n").split("\t")) for line in f.readlines()]
            lines = [(idx, [float(v) for v in vec.split()]) for idx, vec in lines]
            idx_v, vectors = zip(*lines)

    # print("T:", idx_t)
    # print("V:", idx_v)
    assert idx_t == idx_v, "Vectors and text are not aligned."
    
    return [(text, vec) for text, vec in zip(text, vectors)]

def collect_vec(data_path, dwe, Aigt, Aogt, model, punct, rounds = ["first_round", "second_round"]):
    """
    Based on A for the ingroup and the outgroup, collects vectors of the replacements that map to A. 
    Mapping between A and vectors of replacements uses `matcher()`.
    """
    
    igt_vectors = []
    ogt_vectors = []
    
    for meaning in ["ingroup", "outgroup"]:
        if rounds == None:
            for replacement, vector in load_replacements(dwe, meaning, None, model, data_path):
                if meaning == "ingroup":
                    if matcher(replacement, Aigt, punct): # punctuation
                        igt_vectors.append(vector)
                else:
                    if matcher(replacement, Aogt, punct): # punctuation
                        ogt_vectors.append(vector)

        else:
            for rnd in rounds: # ["first_round", "second_round"] or just one of them
                for replacement, vector in load_replacements(dwe, meaning, rnd, model, data_path):
                    if meaning == "ingroup":
                        if matcher(replacement, Aigt, punct): # punctuation
                            igt_vectors.append(vector)
                    else:
                        if matcher(replacement, Aogt, punct): # punctuation
                            ogt_vectors.append(vector)
    
    return np.array(igt_vectors), np.array(ogt_vectors)

def select2vec(mode, dwe, wh_rnds, model, path_dfA, stopwords, punct, data_path, verbose = True):
    """
    Based on a strategy, i.e. `mode`, Select A and returns vectors of replacments that map to A. 
    Uses `select_A()` and `collect_vec()`.
    """
    
    # print("`wh_rnds`:", wh_rnds)
    
    if mode == "rn":    # Really naive; probably the most sensible for SBERT
        
        igt_vectors = []
        if wh_rnds == None:
            _, vecs = zip(*load_replacements(dwe, "ingroup", None, model, data_path))
            igt_vectors.extend(vecs)
        else:
            for rnd in wh_rnds:
                _, vecs = zip(*load_replacements(dwe, "ingroup", rnd, model, data_path))
                igt_vectors.extend(vecs)
        
        ogt_vectors = []
        if wh_rnds == None:
            _, vecs = zip(*load_replacements(dwe, "outgroup", None, model, data_path))
            ogt_vectors.extend(vecs)
        else:
            for rnd in wh_rnds:
                _, vecs = zip(*load_replacements(dwe, "outgroup", rnd, model, data_path))
                ogt_vectors.extend(vecs)  

        # print("I:", igt_vectors[0])
        # print("O:", ogt_vectors[0])
        
        return np.array(igt_vectors), np.array(ogt_vectors)
    
    else:
        
        dfA = pd.read_csv(path_dfA, sep="\t") # check parameters
        dfA = dfA.map(lambda s: s.lower() if type(s) == str else s) # previously .applymap()
        
        if wh_rnds == ["first_round"]:
            PHASE = 1
        elif wh_rnds == ["second_round"]:
            PHASE = 2
        else: # including `wh_rnds == None`
            PHASE = "both"

        # print("PHASE:", PHASE)
        
        if mode == "nno":   # Naive No Overlap
            Aigt, Aogt = select_A(
                df = dfA, 
                dwe = dwe, 
                phase = PHASE, 
                sw = stopwords, 
                punct = punct, 
                empty_intersect = True)

        if mode == "top1":  # Top 1 (no overlap)
            Aigt, Aogt = select_A(
                df = dfA, 
                dwe = dwe,
                phase = PHASE,
                sw = stopwords,
                punct = punct,
                k = 1,
                empty_intersect = True
            )

        if mode == "top3":  # Top 3 (no overlap)
            Aigt, Aogt = select_A(
                df = dfA, 
                dwe = dwe,
                phase = PHASE,
                sw = stopwords,
                punct = punct,
                k = 3,
                empty_intersect = True
            )

        if mode == "ms1":    # Multiple Selection; threshold ... 
            Aigt, Aogt = select_A(
                df = dfA, 
                dwe = dwe,
                phase = PHASE,
                sw = stopwords,
                punct = punct,
                k = 3,
                min_OR = 2.0,
                empty_intersect = False
            )
        
        if verbose:
            if len(Aigt) < 4:
                logging.info(f"Aigt: {', '.join(Aigt)}")
                logging.info(f"Aogt: {', '.join(Aogt)}")
        
        # print("`wh_rnds`:", wh_rnds)
        if wh_rnds == None:
            rounds = None
        elif wh_rnds == 1:
            rounds = ["first_round"]
        elif wh_rnds == 2:
            rounds = ["second_round"]
        else: # i.e. wh_rnds == "both"
            rounds = ["first_round", "second_round"]

        # print("`data_path`:", data_path)
        # print("`dwe`:", dwe)
        # print("`model`:", model)
        # print("`rounds`:", rounds)
        
        return collect_vec(data_path, dwe, Aigt, Aogt, model, punct, rounds)

def angular_distance(v1, v2):
    
    angular = np.arccos(cosine_similarity(v1,v2)) / np.pi # Noble et al
    
    return angular

def PairwiseMeanSimilarity(v, v_list):
    
    pairwise = cosine_similarity(v, v_list)
    pairwise_mean = pairwise.mean()
    
    return pairwise_mean

def find_corpus_vector(dwe, time, path):
    
    # with open(path / model / f"{time}.txt", encoding="utf-8") as f:
    with open(path / f"{time}.txt", encoding="utf-8") as f:
        lines = [line.strip("\n") for line in f.readlines()]
    
    for line in lines:
        term, vector = tuple(line.split("\t"))
        if term == dwe:
            vector = [float(v) for v in vector.split()]
            return np.array(vector)
    
    return None

def similar_string(a, b):
    return SequenceMatcher(None, a, b).ratio()

def repl_dwe(dwe, rule = None, verbose = True):
    
    if rule != None:
        return rule[dwe]
    else: # infer!
        potential_dwes = ["forortsgang", "aterinvandring", "berikar", "globalister"]
        
        dwe = dwe.split("_")[-1]
        
        best_score = 0
        best_guess = None
        
        for candidate in potential_dwes:
            score = similar_string(dwe, candidate)
            if score > best_score:
                best_score = score
                best_guess = candidate
        
        if verbose:
            logging.info(f"Inference for {dwe}: {best_guess} (score = {best_score:.2f}).")
        
        return best_guess

def np2str(arr):
    arr_as_str = " ".join([str(v) for v in arr.tolist()])
    return arr_as_str

def iov_builder(config):

    t0 = time.time()
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO, 
        handlers=[logging.FileHandler(Path(config.log_path), mode= "w"),
                  logging.StreamHandler()]
    )

    for k, v in config.__dict__.items():
        logging.info(f"{k}: {v}")

    config.path_corpus_vectors = Path(config.path_corpus_vectors)
    
    results = []
    centroids = []
    methods = ["I-cnt", "O-cnt", "cnt-ssc", "cnt-smx", "I-pwn", "O-pwn", "pwn-ssc", "pwn-smx"]
    # alternative: make this a config attribute
    # times = [str(t) for t in range(config.first_t, config.last_t+1)]
    TIMES = [int(t.replace(".txt", "")) for t in os.listdir(config.path_corpus_vectors)]
    TIMES.sort()
    TIMES = [int(t) for t in TIMES]
    # print(TIMES)

    if config.stopwords != None:
        with open(Path(config.stopwords), encoding="utf-8") as f:
            stopwords = [w.strip("\n") for w in f.readlines()]
    else:
        stopwords = []
    
    for progress, dwe in enumerate(config.dwes, start = 1):
        t = time.time() - t0
        logging.info(f"PROCESSING {progress} OF {len(config.dwes)}: '{dwe}'; {int(t/60)} m. {int(t%60)} s.")
        dwe_in_replacement_test = repl_dwe(dwe)

        for strategy in config.strategies:
            logging.info(f"Strategy: {strategy}")
            INGROUPvec, OUTGROUPvec = select2vec(
                mode      = strategy, 
                dwe       = dwe_in_replacement_test, 
                wh_rnds   = config.wh_rounds, 
                model     = config.model, 
                path_dfA  = Path(config.dfA_path), 
                stopwords = stopwords, 
                punct     = config.punct, 
                data_path = Path(config.data_path)
            )

            ING_centroid  = INGROUPvec.mean(axis=0)
            OUTG_centroid = OUTGROUPvec.mean(axis=0)

            #print("ING_centroid", ING_centroid)
            #print("OUTG_centroid", OUTG_centroid)            

            if config.save_vectors:
                centroids.append((dwe, strategy, "ingroup", np2str(ING_centroid)))
                centroids.append((dwe, strategy, "outgroup", np2str(OUTG_centroid)))

            if config.full_table:

                d = {method: [] for method in methods}
    
                for TIME in TIMES:
    
                    # if config.model not in os.listdir(config.path_corpus_vectors):
                    #     m_name = config.model.replace("-avg", "") 
                    #     m_name = m_name.replace("-cls", "")
                    #     if m_name not in os.listdir(config.path_corpus_vectors):
                    #         return f"No match of model in among diachronic vectors!! {m_name} not in {', '.join(os.listdir(config.path_corpus_vectors))}"
                    # else:
                    #     m_name = config.model
    
                    vector = find_corpus_vector(dwe, TIME, config.path_corpus_vectors)

                    #print("vector", TIME, dwe, vector)
                    
                    if type(vector) != np.ndarray:
    
                        d["I-cnt"].append(None)
                        d["O-cnt"].append(None)
                        d["cnt-ssc"].append(None)
                        d["cnt-smx"].append(None)
                        d["I-pwn"].append(None)
                        d["O-pwn"].append(None)
                        d["pwn-ssc"].append(None)
                        d["pwn-smx"].append(None)                    
    
                    else:
                        
                        i_cnt = cosine_similarity(vector.reshape(1,-1), ING_centroid.reshape(1,-1))[0][0] 
                        o_cnt = cosine_similarity(vector.reshape(1,-1), OUTG_centroid.reshape(1,-1))[0][0]
    
                        i_pwn = PairwiseMeanSimilarity(vector.reshape(1, -1), INGROUPvec)
                        o_pwn = PairwiseMeanSimilarity(vector.reshape(1, -1), OUTGROUPvec)
                        
                        d["I-cnt"].append(i_cnt)
                        d["O-cnt"].append(o_cnt) 
                        d["cnt-ssc"].append(i_cnt / (i_cnt + o_cnt))
                        d["cnt-smx"].append(softmax([[i_cnt, o_cnt]])[0][0])
    
                        d["I-pwn"].append(i_pwn)
                        d["O-pwn"].append(o_pwn) 
                        d["pwn-ssc"].append(i_pwn / (i_pwn + o_pwn))
                        d["pwn-smx"].append(softmax([[i_pwn, o_pwn]])[0][0])
    
                if config.results_format == "long":
                    
                    for method in d.keys():
                        line = [dwe, strategy, method]
                        line.extend(d[method])
                        results.append(line)
    
                else: # if results_format == "wide"
                    line = [dwe, strategy]
                    for method in d.keys():
                        line.extend(d[method])
                    results.append(line)

    if config.save_vectors:
        pd.DataFrame(centroids, columns=["DWE", "Strategy", "Meaning", "Embedding"]).to_csv(Path(config.IOvec_path), encoding="utf-8", sep="\t")
    
    if config.full_table:
        
        if config.results_format == "long":
            features = ["DWE", "A-Strategy", "Method"] + TIMES
            # if config.add_correlations:
            #     additional_headings = ["r_naive", "r_rect", ...]
            #     features.extend(additional_headings)
            
        
        else: # if wide
            features = ["DWE", "A-Strategy"]
            for method in methods:
                m = [f"{method}_{TIME}" for TIME in TIMES]
                features.extend(m)
        
        df = pd.DataFrame(results, columns = features)
        
        df.to_csv(Path(config.results_path))
    
    t = time.time() - t0
    logging.info(f"Done! {int(t/60)} m. {int(t%60)} s.")

def readJsonConfig(path):
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return json.loads(f.read())

def guess_src(path, shorten=False):
    payload = Path(path).as_posix().split("/")
    model   = payload[-1]
    step    = payload[-2]
    corpus  = payload[-3]

    if shorten:
        model = model.split("-")[0]

    return model, corpus, step

class Config:
    def __init__(self):
        self

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="io_vec.py", description="Build ingroup and outgroup embeddings (measures).")

    parser.add_argument("config", type=str, help="Path to base config file.")
    parser.add_argument("corpus_vec", type=str, help="Path to diachronic corpus vectors. E.g. '../data/diachronic_vectors/fm-smh/monthly/bert-base-swedish-cased/'.")

    parser.add_argument("-p", "--results_path", type=str, help="Path for saving the results.")

    parser.add_argument("-f", "--format", type=str, default="long", help="Format of results. Default: long.")
    parser.add_argument("-t", "--full_table", action="store_false", help="Provide to NOT save the full table.")

    parser.add_argument("-r", "--replacement_root", type=str, help="Root for replacement data. If provided, predefined directories are used for `log_path`, `replacement_data` and `IOvec_path`. Prefix of log-file is inferred if there is no value for `log_name`.")
    parser.add_argument("-m", "--log_name", type=str, help="Prefix for log-file if `--replacement_root`. If not provided, a prefix is contrued from the path `corpus_vec`.")
    parser.add_argument("-l", "--log_path", type=str, help="Path to log.")
    parser.add_argument("-d", "--replacement_data", type=str, help="Path to replacement data.")
    parser.add_argument("-c", "--save_vectors", action="store_true", help="Provide to save ingroup and outgroup vectors.")
    parser.add_argument("-e", "--IOvec_path", type=str, help="Path for saving ingroup and outgroup embeddings. Infered if `--replacement_root`.")

    parser.add_argument("-n", "--rounds", type=str, help="Provide for wave-organised data. What waves/rounds? Separate by space: {first_round, second_round, 'first_round second_round'}. Default: None (i.e. use collapsed data).")


    args = parser.parse_args()

    JSON = readJsonConfig(args.config)

    ROUNDS = args.rounds.split() if args.rounds != None else args.rounds

    if args.replacement_root != None:
        root = Path(args.replacement_root)
        model, corpus, step = guess_src(args.corpus_vec, shorten=True)
        NOW = datetime.datetime.now()
        now_suffix = f"{NOW.year}-{NOW.month}-{NOW.day}--{NOW.hour:02d}-{NOW.minute}"

        DATA     = root / "processed"
        IOVEC    = root / f"io_data/{model}_vecs.csv"

        if args.full_table:
            if args.results_path == None:             
                TABLE = root / f"io_data/{model}_{corpus}_{step}_results.csv"
            else:
                TABLE = args.results_path

        if args.log_name == None:
            LOG_PATH = root / f"log/{model}_{corpus}_{step}_{now_suffix}.log"
        else:
            LOG_PATH = root / f"log/{args.log_name}_{now_suffix}.log"

    else: 
        LOG_PATH = args.log_path
        DATA     = args.replacement_data
        IOVEC    = args.IOvec_path
        TABLE    = args.results_path


    config = Config()

    config.log_path   = LOG_PATH

    config.dwes       = JSON["dwes"]
    config.strategies = JSON["strategies"]
    config.stopwords  = JSON["stopwords"]
    config.punct      = JSON["punct"]
    config.dfA_path   = JSON["dfA_path"]
    config.model      = JSON["model"]

    config.data_path  = DATA   # "../../data/replacements/processed/"
    config.wh_rounds  = ROUNDS # ["first_round", "second_round"]

    config.path_corpus_vectors = args.corpus_vec # i.e. path to diachronic data

    config.results_format = args.format
    config.save_vectors = args.save_vectors # {True, False}
    config.full_table   = args.full_table   # {True, False}
    config.results_path = TABLE 
    config.IOvec_path   = IOVEC             # "../../data/replacements/io_data/bert_io_vecs.csv"    #../data/replacements/results/mt5-xl_IOvec.csv")


    for k, v in config.__dict__.items():
        print(k, ":", v)

    iov_builder(config)

    print()
    print("Done!")                