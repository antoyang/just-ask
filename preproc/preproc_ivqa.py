import os
import collections
import json
import pandas as pd

from global_parameters import IVQA_PATH

os.chdir(IVQA_PATH)

train_df = pd.read_csv("train.csv")


def get_vocabulary(train_df, save=False):
    train_counter = collections.Counter(
        list(train_df["answer1"])
        + list(train_df["answer2"])
        + list(train_df["answer3"])
        + list(train_df["answer4"])
        + list(train_df["answer5"])
    )
    most_common = train_counter.most_common()
    vocab = {}
    for i, x in enumerate(most_common):  # 2349 answers present twice
        if x[1] >= 2:
            vocab[x[0]] = i
        else:
            break
    print(len(vocab))
    if save:
        with open("vocab.json", "w") as outfile:
            json.dump(vocab, outfile)
    return vocab


vocab = get_vocabulary(train_df, True)
