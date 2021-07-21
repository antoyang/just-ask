import json
import os
import collections
import pandas as pd

from global_parameters import ACT_PATH

os.chdir(ACT_PATH)

train_q = json.load(open("train_q.json", "r"))
val_q = json.load(open("val_q.json", "r"))
test_q = json.load(open("test_q.json", "r"))

train_a = json.load(open("train_a.json", "r"))
val_a = json.load(open("val_a.json", "r"))
test_a = json.load(open("test_a.json", "r"))


def get_vocabulary(train_a, save=False):
    ans = [x["answer"] for x in train_a]
    train_counter = collections.Counter(ans)
    most_common = train_counter.most_common()
    vocab = {}
    for i, x in enumerate(most_common):  # 1654 answers present twice
        if x[1] >= 2:
            vocab[x[0]] = i
        else:
            break
    print(len(vocab))
    if save:
        with open("vocab.json", "w") as outfile:
            json.dump(vocab, outfile)
    return vocab


def json_to_df(vocab, train_q, train_a, val_q, val_a, test_q, test_a, save=False):
    # Verify alignment of files
    for x, y in zip(train_q, train_a):
        assert x["question_id"] == y["question_id"]
    for x, y in zip(val_q, val_a):
        assert x["question_id"] == y["question_id"]
    for x, y in zip(test_q, test_a):
        assert x["question_id"] == y["question_id"]

    train_df = pd.DataFrame(
        {
            "question": [x["question"] for x in train_q],
            "answer": [x["answer"] for x in train_a],
            "video_id": [x["video_name"] for x in train_q],
            "type": [x["type"] for x in train_a],
        },
        columns=["question", "answer", "video_id", "type"],
    )
    print(len(train_df))
    train_df = train_df[
        train_df["answer"].isin(vocab)
    ]  # do not use train samples of which the answer is not in the vocab
    val_df = pd.DataFrame(
        {
            "question": [x["question"] for x in val_q],
            "answer": [x["answer"] for x in val_a],
            "video_id": [x["video_name"] for x in val_q],
            "type": [x["type"] for x in val_a],
        },
        columns=["question", "answer", "video_id", "type"],
    )
    test_df = pd.DataFrame(
        {
            "question": [x["question"] for x in test_q],
            "answer": [x["answer"] for x in test_a],
            "video_id": [x["video_name"] for x in test_q],
            "type": [x["type"] for x in test_a],
        },
        columns=["question", "answer", "video_id", "type"],
    )

    print(len(train_df), len(val_df), len(test_df))

    if save:
        train_df.to_csv("train.csv", index=False)
        val_df.to_csv("val.csv", index=False)
        test_df.to_csv("test.csv", index=False)

    return train_df, val_df, test_df


vocab = get_vocabulary(train_a, True)
train_df, val_df, test_df = json_to_df(
    vocab, train_q, train_a, val_q, val_a, test_q, test_a, True
)
