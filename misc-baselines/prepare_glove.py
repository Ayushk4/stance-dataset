# Ip: glove/smaller.json
# Op: glove/prepared.json

# Adds to vocab, the tokens with more than 10 occurrences
# Handle special tokens.

import argparse
import json
from collections import Counter
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--glove_dims", default=200, type=int)
parser.add_argument("--min_occur", default=10, type=int)
params = parser.parse_args()

DATASET = "../dataset/data_new.json"
GLOVE_SMALLER_PATH = "glove/smaller.json"
SAVE_PATH = "glove/prepared.json"

def get_dataset_counts():
    fo = open(DATASET, "r")
    dataset = json.load(fo)
    fo.close()

    all_tokens = []
    for d in dataset:
        all_tokens.extend([x.lower() for x in d["text"]])
        all_tokens.extend([x.lower() for x in d["target"].split(' ')])
    counts = Counter(all_tokens)

    return counts

ALL_TOKENS = get_dataset_counts()

fo = open(GLOVE_SMALLER_PATH, "r")
glove = json.load(fo)
fo.close()

print(len(ALL_TOKENS), len(glove.keys()))

special_tokens = ["<number>", "<user>", "<money>", "<pad>"]


for tok in ALL_TOKENS:
    if ALL_TOKENS[tok] > params.min_occur and \
        (tok in special_tokens or tok not in glove.keys()): # We reinitialize special tokens
        if tok == "<pad>": # We don't need embedding for pad
            print(glove.pop(tok, "  pad not present"))
        else:
            glove[tok] = list(np.random.uniform(low=-0.1, high=0.1, size=params.glove_dims))
            # print(tok, "\t", ALL_TOKENS[tok], "\t", "%.4f" % glove[tok][0])

print(len(ALL_TOKENS), len(glove.keys()))

glove["<unknown>"] = list(np.random.uniform(low=-0.1, high=0.1, size=params.glove_dims))

fo = open(SAVE_PATH, "w+")
json.dump(glove, fo)
fo.close()