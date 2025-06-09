
import torch
import os
import numpy as np
from pathlib import Path
import time
import pandas as pd
import argparse

def naming(full_path):
    return full_path.split("/")[-1]

def txt2vec(data, tokenizer, model, pooling, device, sentence_transformer):
    t = time.time()
    print()
    vectors = []
    for i, (idx, line) in enumerate(zip(data.index, data.iloc[:,0]), start=1):
        #print("Line:", line)
        pcent = round((i / data.shape[0]) * 100, 1)
        print(f"{pcent:<10}{int((time.time()-t))} s.", end="\r")

        if sentence_transformer:
            vector = model.encode(line)

        else: 
            ###################  Regular BERT  #######################################
            encoded = tokenizer.encode_plus(line, return_tensors="pt", truncation=True, max_length=512)
            encoded.to(device)
            with torch.no_grad():
                output = model(**encoded)
            last_hidden = output.last_hidden_state.squeeze()

            if pooling == "cls":
                vector = last_hidden[0] # CLS
            elif pooling == "avg":
                vector = torch.mean(last_hidden, dim=0)
            else:
                return "No pooling method!!"
            ##########################################################################

        as_str = " ".join([str(value) for value in vector.tolist()])
        vectors.append(f"{idx}\t{as_str}\n") # <-- Note: "\n" in string

    return vectors

def vec2file(vectors, path):
    with open(path, encoding="utf-8", mode = "w") as f:
        for vec in vectors:
            f.write(vec) # "\n" already in string

def text2bert(data_path, local_models_at, data_struct, huggface_models = [], sentence_transformer=False, device = "cpu", pooling = "cls", use_suffix = False):

    if sentence_transformer:
        from sentence_transformers import SentenceTransformer
    else:
        from transformers import AutoModel, AutoTokenizer
    
    t0 = time.time()
    if use_suffix:
        suffix = f"-{pooling}"
    else:
        suffix = ""
        
    data_path = Path(data_path)

    if local_models_at == None:
        models = huggface_models
    else:
        models_path = Path(local_models_at) # ???
        models = [f"{local_models_at}/{model}" for model in os.listdir(local_models_at)] + huggface_models

    print("Models:", models)

    for model in models:
        
        if sentence_transformer:
            tokenizer = None
            BERT = SentenceTransformer('KBLab/sentence-bert-swedish-cased')

        else:
            #############  Regular BERT  ###########################
            tokenizer = AutoTokenizer.from_pretrained(model)
            BERT = AutoModel.from_pretrained(model)
            ########################################################
        
        BERT.to(device)

        for dwe in os.listdir(data_path):
            for meaning in ["ingroup", "outgroup"]:

                if data_struct == "two-wave":
                    for rnd in ["first_round", "second_round"]:
                        print(f"{dwe:<15}{meaning:<10}{rnd:<15}{naming(model)}")

                        replacements = pd.read_csv(data_path / dwe / meaning / rnd / "replacements.txt", sep = "\t", index_col = 0, header=None)
                        vectors = txt2vec(
                            data      = replacements, 
                            tokenizer = tokenizer, 
                            model     = BERT, 
                            pooling   = pooling,
                            device    = device,
                            sentence_transformer = sentence_transformer
                        )
                        
                        path_for_vec = data_path / dwe / meaning / rnd / "vectors" / f"{naming(model)}{suffix}"
                        isExist = os.path.exists(path_for_vec)
                        if not isExist:
                            os.makedirs(path_for_vec)
                        vec2file(vectors, path = path_for_vec / "vecs.txt")

                elif data_struct == "combined":
                    print(f"{dwe:<15}{meaning:<10}{naming(model)}")
                    
                    replacements = pd.read_csv(data_path / dwe / meaning / "replacements.txt", sep = "\t", index_col = 0, header=None)
                    vectors = txt2vec(
                        data      = replacements, 
                        tokenizer = tokenizer, 
                        model     = BERT, 
                        pooling   = pooling,
                        device    = device,
                        sentence_transformer = sentence_transformer
                    )
                    
                    path_for_vec = data_path / dwe / meaning / "vectors" / f"{naming(model)}{suffix}"
                    isExist = os.path.exists(path_for_vec)
                    if not isExist:
                        os.makedirs(path_for_vec)
                    vec2file(vectors, path = path_for_vec / "vecs.txt")

                else:
                    print(f"`data_struct` must be 'two-wave' or 'combined'.")
                
    print()
    print("Done!", int((time.time()-t0)/60), "m.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="bert_repl_vec.py", description = 'Builds word vectors of replacement data.')

    parser.add_argument('data_path', type=str, help='Path to replacement data (root)')
    parser.add_argument('data_struct', type=str, choices=["two-wave", "combined"], help='Structure of data. Use `two-wave` for two wave structure. Use `combined` for combined, or collapsed, data structure.')
    parser.add_argument('-m', '--local_model', type=str, help='Path to local model(s).')
    parser.add_argument('-f', '--huggingface_model', type=str, default='KBLab/bert-base-swedish-cased', help="Hugging Face model. Default: KBLab/bert-base-swedish-cased.")
    parser.add_argument('-t', '--sentence_transformer', action='store_true', help="Provide if model is sentence transformer.")
    parser.add_argument('-d', '--device', type=str, choices=["cuda", "cpu"], default="cpu", help="Torch `device`: `cuda` or `cpu`. Default: `cpu`.")
    parser.add_argument('-p', '--pooling', type=str, choices=["cls", "avg"], default="cls", help="Principle for averaging embeddings of words in replacements (not applicable to sentence transformers): `cls` for using `[CLS]` token (default); `avg` for taking the average of the token embeddings. Default: `cls`.")
    parser.add_argument('-s', '--use_suffix', action="store_true", help="Provide to add `pooling` parameter in naming output. Default: do not add suffix.")
    args = parser.parse_args()

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")

    text2bert(
        data_path            = args.data_path, 
        local_models_at      = args.local_model, 
        data_struct          = args.data_struct, 
        huggface_models      = [args.huggingface_model], # note list (!?)
        sentence_transformer = args.sentence_transformer,
        device               = args.device, 
        pooling              = args.pooling, 
        use_suffix           = args.use_suffix
        )   
