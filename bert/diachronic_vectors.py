import numpy as np
from pathlib import Path
import os
import argparse

# Merger. For compounds. 
merger = {
    "N1C_berikareX": "N1_berikare",
    "N1C_förortsgängX": "N1_förortsgäng",
    "N1C_globalistX": "N1_globalist",
    "N1C_kulturberikarX": "N1_kulturberikare",
    "N1C_återvandringsX": "N1_återvandring",
    "N2C_återvandrarX": "N2_återvandrare",
}

def file2vec(path, merge):
    with open(path, encoding="utf-8") as f:
        data = [tuple(line.strip("\n").split("\t")) for line in f.readlines()]
        data = [(term, [float(v) for v in vec.split()]) for term, vec in data]
    d = dict()
    for term, vec in data:
        if merge:
            if term in merger:
                term = merger[term]
        if term in d:
            d[term].append(vec)
        else:
            d[term] = [vec]

    centroids = {term: np.array(vec).mean(axis=0) for term, vec in d.items()}
    return centroids

def main(token_vectors_dir, output_dir, merge):
    token_vectors_dir = Path(token_vectors_dir)
    output_dir = Path(output_dir)
    files = os.listdir(token_vectors_dir)
    for t_file in files:
        centroids = file2vec(token_vectors_dir / t_file, merge)
        with open(output_dir / t_file, encoding = "utf-8", mode = "w") as f:
            for term, centroid in centroids.items():
                to_file = f"{term}\t{' '.join([str(v) for v in centroid.tolist()])}\n"
                f.write(to_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="diachronic_vectors.py", description="Get average vectors from token vectors at time t1, ... tn.")
    parser.add_argument("tok_vec_dir", type=str, help="Path to directory with token vectors for t1, ..., tn.")
    parser.add_argument("output_dir", type=str, help="Path to directory for the average vectors for t1, ..., tn (output).")
    parser.add_argument("-m", "--use_merger", action="store_true", help="Provide to merge compounds. Deafult is to merge.")

    args = parser.parse_args()

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")

    print("Start process.")

    main(args.tok_vec_dir, args.output_dir, args.use_merger)

    print("Done!")