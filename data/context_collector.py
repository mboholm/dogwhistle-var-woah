import re
from pathlib import Path
import os
import time
import argparse

def read_util(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        terms = [line.strip("\n").split("#")[0] for line in f.readlines()]
    terms = [term for term in terms if term != ""]
    return terms 

def read_paradigm(file_path):
    paradigms = read_util(file_path)
    paradigms = [tuple([re.sub(r" \Z", r"", column) for column in p.split(" -> ")]) for p in paradigms]
    return paradigms

def parser(line, hit, paradigm):
    
    lemmas = []
    
    for pos, regex, lemma in paradigm:
        regex = re.compile(regex)
        if re.search(regex, line) != None:
            cls_name = f"{pos}_{lemma}"
            lemmas.append(cls_name)
    
    lemmas = lemmas if lemmas != [] else [f"X_{hit.group()}"]
    
    return "{}\t{}\t{}".format("; ".join(lemmas), len(lemmas), line)

def context_collector(corpus_in, corpus_out, full_paradigm, stop=None):
    
    t0 = time.time()
    corpus_in = Path(corpus_in)
    corpus_out = Path(corpus_out)
    files = sorted(os.listdir(corpus_in))
    
    roots, pos, regex, lemmas = zip(*read_paradigm(full_paradigm))

    paradigm = list(zip(pos, regex, lemmas))
    roots = set(roots)
    roots = re.compile(r"(" + "|".join(roots) + ")")

    for k, file in enumerate(files, start=1):

        if stop != None:
            if k == stop:
                return        

        f_out = open(corpus_out / file, "w", encoding="utf-8")
        f_out.close()

        with open(corpus_in / file, "r", encoding="utf-8") as f_in, open(corpus_out / file, "a", encoding="utf-8") as f_out:
            year = file.strip(".txt")

            for i, line in enumerate(f_in):

                if i % 10000 == 0:
                    print(f"PROCESSED INPUT: {file} {k} / {len(files)}: {i:>10}", end="\r")

                
                hit = re.search(roots, line) 
                if hit == None:
                    continue 
                    
                # get lemma(s), line "LEM    EXAMPLE"
                line = line.strip("\n")
                to_write = parser(line, hit, paradigm)

                f_out.write(to_write + "\n")
    
    delta_t = time.time() - t0
    m = int(delta_t / 60)
    s = int(delta_t / 60)
    
    print()
    print("Done!", f"({m} m, {s} s)")             
    
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("corpus_in", type=str, help="path to corpus to collect from")
    argparser.add_argument("corpus_out", type=str, help="path to output directory")
    argparser.add_argument("paradigm", type=str, help="path to paradigm file")
    argparser.add_argument("--stop", "-s", type=int, help="stop at this file (order)")

    args = argparser.parse_args()

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")    
    
    context_collector(args.corpus_in, args.corpus_out, args.paradigm, args.stop)



