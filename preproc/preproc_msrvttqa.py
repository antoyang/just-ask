import json
import os
import pandas as pd
import collections

from global_parameters import MSRVTT_PATH

os.chdir(MSRVTT_PATH)

train_json = json.load(open("train_qa.json", "r"))
val_json = json.load(open("val_qa.json", "r"))
test_json = json.load(open("test_qa.json", "r"))
types = {"what": 0, "how": 1, "color": 2, "where": 3, "who": 4, "when": 5}


def get_vocabulary(train_json, save=False):
    train_counter = collections.Counter([x["answer"] for x in train_json])
    most_common = train_counter.most_common(4000)  # top 4K answers
    vocab = {}
    for i, x in enumerate(most_common):
        vocab[x[0]] = i
    print(len(vocab))
    if save:
        with open("vocab.json", "w") as outfile:
            json.dump(vocab, outfile)
    return vocab


def get_type(question):
    if "color" in question:
        return types["color"]
    elif question.split(" ")[0] in ["what", "who", "where", "when", "how"]:
        return types[question.split(" ")[0]]
    else:
        raise NotImplementedError


def json_to_df(vocab, train_json, val_json, test_json, save=False):
    train_df = pd.DataFrame(
        {
            "question": [x["question"] for x in train_json],
            "answer": [x["answer"] for x in train_json],
            "category_id": [x["category_id"] for x in train_json],
            "video_id": [x["video_id"] for x in train_json],
            "id": [x["id"] for x in train_json],
        },
        columns=["question", "answer", "category_id", "video_id", "id"],
    )
    print(len(train_df))
    train_df = train_df[
        train_df["answer"].isin(vocab)
    ]  # do not use train samples of which the answer is not in the vocab
    val_df = pd.DataFrame(
        {
            "question": [x["question"] for x in val_json],
            "answer": [x["answer"] for x in val_json],
            "category_id": [x["category_id"] for x in val_json],
            "video_id": [x["video_id"] for x in val_json],
            "id": [x["id"] for x in val_json],
        },
        columns=["question", "answer", "category_id", "video_id", "id"],
    )
    test_df = pd.DataFrame(
        {
            "question": [x["question"] for x in test_json],
            "answer": [x["answer"] for x in test_json],
            "category_id": [x["category_id"] for x in test_json],
            "video_id": [x["video_id"] for x in test_json],
            "id": [x["id"] for x in test_json],
        },
        columns=["question", "answer", "category_id", "video_id", "id"],
    )

    train_df["type"] = [get_type(x) for x in train_df["question"]]
    val_df["type"] = [get_type(x) for x in val_df["question"]]
    test_df["type"] = [get_type(x) for x in test_df["question"]]

    print(len(train_df), len(val_df), len(test_df))

    if save:
        train_df.to_csv("train.csv", index=False)
        val_df.to_csv("val.csv", index=False)
        test_df.to_csv("test.csv", index=False)

    return train_df, val_df, test_df


vocab = get_vocabulary(train_json, True)
train_df, val_df, test_df = json_to_df(vocab, train_json, val_json, test_json, True)
