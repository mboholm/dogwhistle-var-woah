import re
from pathlib import Path
import os
import time
from string import punctuation
import json
import argparse

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

punctuation = punctuation.replace(" ", "") # white space is in punctuation
#print(type(punctuation))

def preprocess1(example,
                remove_stopwords = False,
                stopwords = None,
                remove_numbers = False,
                remove_urls = True,
                remove_punctuations = False,
                min_tok_utterances = None,
                lower = True,
                remove_emojis = True):
    """ 
    Preprocess a string.
    """

    example = example.split()
    
    if remove_stopwords:
        assert isinstance(stopwords, list) or isinstance(stopwords, set), "You have selected to use stopwords, but no stopword list is provided."

        example = [word for word in example if word not in stopwords]
        
#     if remove_punctuations:
#         example = [token for token in example if not token in punctuation]
        
    if min_tok_utterances != None:
        if len(example) <= min_tok_utterances:
            return ""
        
    example = " ".join(example)
    
    if lower:
        example = example.lower()
    
    if remove_numbers:
        example = re.sub(r"[0-9]+", "", example)
    
    if remove_urls:
        example = re.sub(r"https?://.*", "", example)
        example = re.sub(r"www\..*", "", example)
        
    if remove_emojis:
        example = re.sub(emoji_pattern, "", example)

    if remove_punctuations:
        example = example.translate(str.maketrans('', '', punctuation))
        #example = re.sub(re.compile(r"[^\w\s]"), "", example)       

    return example

def preprocess2(corpus_in, corpus_out, CONFIG):
    """
    Preprocess a corpus.
    """
    
    t0 = time.time()
    corpus_in = Path(corpus_in)
    corpus_out = Path(corpus_out)
    files = sorted(os.listdir(corpus_in))

    for k, file in enumerate(files, start=1):
        
        f_out = open(corpus_out / file, "w")
        f_out.close()
        
        with open(corpus_in / file, "r", encoding="utf-8") as f_in, open(corpus_out / file, "a", encoding="utf-8") as f_out:
            year = file.strip(".txt")
            
            for i, line in enumerate(f_in):
                
#                 if k > 2:
#                     return 
                
                if i % 10000 == 0:
                    print(f"{file} {k} / {len(files)}: {i}    ", end="\r")
                
                if CONFIG != None:
                    line = preprocess1(line, 
                                       remove_stopwords = CONFIG["remove_stopwords"],
                                       stopwords = CONFIG["stopwords"],
                                       remove_numbers = CONFIG["remove_numbers"],
                                       remove_urls = CONFIG["remove_urls"],
                                       remove_punctuations = CONFIG["remove_punctuations"],
                                       min_tok_utterances = CONFIG["min_tok_utterances"],
                                       lower = CONFIG["lower"],
                                       remove_emojis = CONFIG["remove_emojis"])
                else:
                    preprocess1(line) # Use defaults

                if line.strip("\n") == "":
                    continue
                
                line = re.sub(r"-+", "", line)
#                 line = re.sub(r"\.\.+", "", line)
#                 line = re.sub(":", "", line)
#                 line = re.sub(",", "", line)
                line = re.sub(r"  +", " ", line)
                
                # Some language recognition? Only include Swedish
                
                f_out.write(line + "\n")

    
    delta_t = time.time() - t0
    m = int(delta_t / 60)
    s = int(delta_t / 60)
    
    print()
    print("Done!", f"({m} m, {s} s)")             

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_in", type=str, help="path to corpus to be prepocessed")
    parser.add_argument("corpus_out", type=str, help="path to (preprocessed) output")
    parser.add_argument("--config", type=str, help="path to config file")


    args = parser.parse_args()

    if args.config == None:
        preprocess2(corpus_in, corpus_out) # Uses default values
    else:
        with open(Path(args.config), "r", encoding="utf-8") as f:
            config = json.loads((f.read()))
        print("CONFIG:", json.dumps(config, indent=4))

        preprocess2(Path(args.corpus_in), Path(args.corpus_out), config)
