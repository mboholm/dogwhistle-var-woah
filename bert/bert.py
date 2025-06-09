import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
from pathlib import Path
from difflib import SequenceMatcher
import os
#from transformers import pipeline
#from transformers import AutoModel, AutoTokenizer
import time
import argparse

# Ad-hoc rules for handling tokenisation
def map_tok(sentence):
    # sentence = re.sub(r" ([\.,!?])", r"\1", sentence)
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("+", " + ")
    sentence = sentence.replace("&", " & ")
    sentence = sentence.replace(":", " : ")
    sentence = sentence.replace("*", " * ")
    sentence = sentence.replace("^", " ^ ")
    sentence = sentence.replace("ü", "u")
    sentence = sentence.replace("$", "s")
    # sentence = sentence.replace(">", "")
    # sentence = re.sub(r"([a-zåäö]):([a-zåäö])", r"\1 : \2", sentence)
    sentence = sentence.replace("=) ", "")
    sentence = sentence.replace(">= ", "")
    sentence = sentence.replace("=>", "")
    sentence = sentence.replace(">>", ' " ')
    sentence = sentence.replace("<<", ' " ')
    sentence = sentence.replace("| ", "")
    sentence = re.sub(r"([a-zåäö])(['`])([a-zåäö])", r"\1 \2 \3", sentence)
    sentence = re.sub(r"([)\?=%!<>~«])([a-zåäö0-9])", r"\1 \2", sentence)
    sentence = re.sub(r"([a-zåäö0-9])([\))\?=%!<>~«])", r"\1 \2", sentence)
    sentence = re.sub(r"([0-9]),([0-9])", r"\1 , \2", sentence)
    sentence = re.sub(r"#+", " * ", sentence)
    sentence = re.sub(r"([=¤])+", r"\1", sentence)
    sentence = re.sub(r" +", " ", sentence)
    return sentence


def p_split(string):
    k, v = (f"{string.split(' -> ')[1]}_{string.split()[-1]}", string.split(' -> ')[2])
    return k,v

def get_word_vector(sentence, exact_match, lemma, model, tokenizer, device = "cpu", only_check = True):
    if lemma.startswith("X"): # X_globalist
        true_lemma = lemma.split("_")[-1] 
        true_wf = true_lemma + exact_match.split(true_lemma)[-1]
    else: 
        if lemma.endswith("X"): # N1C_globalistX
            true_lemma = lemma.split("_")[1][:-1]
            true_wf = lemma.split("_")[1][:-1] #true_lemma
        else: # N1_globalist
            true_lemma = lemma.split("_")[1]
            true_wf = exact_match

    # sentence=loop_for_idx(sentence)
    encoded = tokenizer.encode_plus(sentence, return_tensors="pt", truncation=True, max_length=512)

    # Check if exact match in  part of sentence accepted by max length:
    max_idx = max([idx for idx in encoded.word_ids() if idx != None])

    if len(sentence.split()) > max_idx: 
        try:
            match_idx = sentence.split().index(exact_match)
        except:
            return # exact match not in sentence
        sentence = " ".join(sentence.split()[match_idx-50:]) 
        encoded = tokenizer.encode_plus(sentence, return_tensors="pt", truncation=True, max_length=512)

        #while exact_match not in sentence.split()[:max_idx+1]:
        #    print("re-define sentence...")
        #    sentence = " ".join(sentence.split()[max_idx:])
        #    encoded = tokenizer.encode_plus(sentence, return_tensors="pt", truncation=True, max_length=512)
        #    max_idx = max([idx for idx in encoded.word_ids() if idx != None])
        #if exact_match not in sentence.split()[:max_idx+1]:
        #    sentence = " ".join(sentence.split()[max_idx:])
        #    encoded = tokenizer.encode_plus(sentence, return_tensors="pt", truncation=True, max_length=512)
    #print(encoded)

    tokens = [tokenizer.decode(wid) for wid in encoded["input_ids"][0]]

    try:
        idx = sentence.split().index(exact_match) # will not match tokenizer; hence `map_tok()`
    except:
        #print("Oops! `exact_match` not in sentence.", lemma, exact_match, sentence)
        return
    
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)[0]
    #print(token_ids_word)
    
    # Handle complex compounds:
    if lemma.endswith("X") or lemma.startswith("X"):
        start_with = min(token_ids_word)
        outer      = start_with
        top        = 0

        if lemma.startswith("X"):
            for i, idx in enumerate(token_ids_word):
                if true_lemma.startswith(tokens[idx].replace("##", "")):
                    token_ids_word = token_ids_word[i:]
                    start_with = min(token_ids_word)
                    break

        for i in token_ids_word: 
            i = i + 1
            candidate = "".join([tok.replace("##", "") for tok in tokens[min(token_ids_word):i]])
            score = SequenceMatcher(None, true_wf, candidate).ratio()
            if score >= top:
                top = score
                outer = i

        token_ids_word = np.arange(start_with, outer) 
        #print(token_ids_word)

    if only_check:
        #print(" ".join(tokens))#, end = "\r")
#        try:
#            token_ids_word[0]
#        except:
#            print("idx -->", idx)
#            print("Tokens -->", f">>>{len(tokens)}<<<", tokenizer.convert_tokens_to_string(tokens))
#            print("max idx covered -->", max([v for v in encoded.word_ids() if v != None]))
#            print("Token ids -->", token_ids_word)
#            print("Sent -->",sentence)
#            print("Exact match -->",exact_match)
        tokens = tokens[token_ids_word[0]:token_ids_word[-1]+1]
        tokens_short = "".join([tok.replace("##", "") for tok in tokens])

        return

    encoded.to(device)
    with torch.no_grad():
        output = model(**encoded)

    last_hidden = output.last_hidden_state.squeeze()
    word_tokens_output = last_hidden[token_ids_word]

    return word_tokens_output.mean(dim=0)

def get_word_embeddings(
    model,
    tokenizer,
    directory,
    vector_dir,
    paradigms,
    ignore,
    device="cpu",
    only_check = True,
    re_start = None
):

    directory = Path(directory)
    vector_dir = Path(vector_dir)
    files = os.listdir(directory)
    if re_start != None:
        times = sorted([int(y.replace(".txt", "")) for y in files])
        files = [f"{y}.txt" for y in times if y >= re_start]
    
    # files = ["2001.txt"]

    model.to(device)

    for file in files:
        print()
        print(file)
        with open(directory / file, encoding="utf-8") as f, open(vector_dir / file, "w", encoding="utf-8") as out:
            for i, line in enumerate(f):
                if i % 10 == 0:
                    print(i, end="\r")
                lemma, n, sentence = tuple(line.strip("\n").split("\t"))

                sentence = map_tok(sentence) # remove tokenization mismatch issues

                if int(n) == 1: 
                    if lemma.startswith(ignore):
                        continue
                    if lemma in paradigms:
                        regex = paradigms[lemma]
                        regex = re.compile(regex)    
                        exact_match = re.search(regex, sentence)
                        if exact_match == None:
                            print("ERROR:", lemma, "||", sentence)
                        exact_match = exact_match.group()                        
                    else:
                        regex = re.compile(f"\\b[0-9a-zåäö]*{lemma.split('_')[-1]}.*?\\b")
                        exact_match = re.search(regex, sentence)
                        if exact_match == None:
                            print("ERROR:", lemma, "|", regex, "|", sentence)
                        exact_match = exact_match.group()
                    vector = get_word_vector(sentence, exact_match, lemma, model, tokenizer, device, only_check)
                    if only_check or vector == None:
                        continue
                        
                    vector = " ".join([str(v) for v in vector.tolist()]) # consider torch.save(tensor, 'file.pt')
                    out.write(f"{lemma}\t{vector}\n")                   

                else: # two instances of the same lemma = problem
                    for l in lemma.split("; "):
                        if l.startswith(ignore):
                            continue                        
                        if l in paradigms:
                            regex = paradigms[l]
                            regex = re.compile(regex)    
                            exact_match = re.search(regex, sentence).group() 
                        else:
                            regex = re.compile(f"\\b[0-9a-zåäö]*{l.split('_')[-1]}.*?\\b")
                            exact_match = re.search(regex, sentence).group()
                        vector = get_word_vector(sentence, exact_match, l, model, tokenizer, device, only_check)
                        if only_check or vector == None:
                            continue
                        vector = " ".join([str(v) for v in vector.tolist()])
                        out.write(f"{l}\t{vector}\n")

def get_sentence_embeddings(
    model,
    #tokenizer,
    directory,
    vector_dir,
    #paradigms,
    ignore,
    device="cpu",
    only_check = True,
    re_start = None
):

    directory = Path(directory)
    vector_dir = Path(vector_dir)
    files = os.listdir(directory)
    if re_start != None:
        times = sorted([int(y.replace(".txt", "")) for y in files])
        files = [f"{y}.txt" for y in times if y >= re_start]
    
    # files = ["2001.txt"]

    model.to(device)

    for file in files:
        print()
        print(file)
        with open(directory / file, encoding="utf-8") as f, open(vector_dir / file, "w", encoding="utf-8") as out:
            for i, line in enumerate(f):
                if i % 10 == 0:
                    print(i, end="\r")
                lemma, n, sentence = tuple(line.strip("\n").split("\t"))

                sentence = map_tok(sentence) # remove tokenization mismatch issues (mainly for index matching issues with BERT)

                if int(n) == 1: 
                    if lemma.startswith(ignore):
                        continue

                    vector = model.encode(sentence)
                        
                    vector = " ".join([str(v) for v in vector.tolist()]) # consider torch.save(tensor, 'file.pt')
                    out.write(f"{lemma}\t{vector}\n")                   

                else: # two instances of the same lemma = problem
                    for l in lemma.split("; "):
                        if l.startswith(ignore):
                            continue     

                        vector = model.encode(sentence)
                        
                        vector = " ".join([str(v) for v in vector.tolist()])
                        out.write(f"{l}\t{vector}\n")

def main(args):
    t0 = time.time()

    if args.sentence_transformer:
        from sentence_transformers import SentenceTransformer

        model_name = args.model
        MODEL = SentenceTransformer(model_name)
        short_name = model_name.split("/")[-1]
        print(short_name) 

        get_sentence_embeddings(
            model      = MODEL,
            directory  = args.txt_dir, 
            vector_dir = args.vec_dir,
            ignore     = tuple(["FooBar_"]),
            device     = args.device,#"cuda",
            only_check = args.only_check,
            re_start   = args.re_start
        )
    
    else: # i.e. model is not a sentence transformer
        from transformers import AutoModel, AutoTokenizer

        with open(Path(args.paradigms), encoding="utf-8") as f:
            paradigms = [line.strip("\n") for line in f.readlines() if not line.startswith("#")]
        paradigms = [p for p in paradigms if p != ""]
        paradigms = [p.split(" #")[0] for p in paradigms]
        paradigms = dict([p_split(paradigm) for paradigm in paradigms])
        
        for k, v in paradigms.items():
            print(k, ":", v)        

        model_name = args.model # 'KB/bert-base-swedish-cased' 
        TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        MODEL = AutoModel.from_pretrained(model_name)
        
        short_name = model_name.split("/")[-1]
        print(short_name)

        get_word_embeddings(
            model      = MODEL,  
            tokenizer  = TOKENIZER, 
            directory  = args.txt_dir, 
            vector_dir = args.vec_dir,
            paradigms  = paradigms,
            ignore     = tuple(["FooBar_"]),
            device     = args.device,#"cuda",
            only_check = args.only_check,
            re_start   = args.re_start
        )

    t = time.time()-t0
    print("Done!", int(t/60), "m.", int(t%60))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="bert.py", description="Get BERT embeddings for terms in context.")
    parser.add_argument("txt_dir", type=str, help="Path to directory with temporally indexed text-files.")
    parser.add_argument("vec_dir", type=str, help="Path to directory for the vectors (output).")
    parser.add_argument("device", type=str, choices=["cuda", "cpu"], help="Device: `cuda` or `cpu`.")
    parser.add_argument("-p", "--paradigms", type=str, help="Path to paradigm file (regex for inflectional forms). Required for index matching, but not for sentence transformer.")
    parser.add_argument("-m", "--model", type=str, default='KB/bert-base-swedish-cased', help="Pre-trained model. Default: KB/bert-base-swedish-cased (huggingface).")
    parser.add_argument("-t", "--sentence_transformer", action="store_true", help="Provide if model is sentence transformer, i.e. no index matching for word embedding is required")
    parser.add_argument("-c", "--only_check", action='store_true', help="Provide to only check tokenisation with regard to the index of the Dogwhistle Term.")
    parser.add_argument("-s", "--re_start", type=int, help="Provide to re-start at this time.")

    args = parser.parse_args()

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")

    main(args)

    print()
    print("Done!")