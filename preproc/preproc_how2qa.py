from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from global_parameters import HOW2QA_PATH, SSD_DIR

train_csv = pd.read_csv(os.path.join(HOW2QA_PATH, "how2QA_train_release.csv"))
train_csv.columns = ["vid_id", "timesteps", "a2", "a3", "a4", "question", "a1"]
print(len(train_csv))
val_csv = pd.read_csv(os.path.join(HOW2QA_PATH, "how2QA_val_release.csv"))
val_csv.columns = ["vid_id", "timesteps", "a2", "a3", "a4", "question", "a1"]
print(len(val_csv))

count = {}
path = os.path.join(SSD_DIR, "s3d_features", "howto100m_s3d_features")

def process(df):
    idx = [[1, 2, 3, 4] for _ in range(len(df))]
    for i in range(len(idx)):
        np.random.shuffle(idx[i])
    ids, a1, a2, a3, a4, answer, question, starts, ends = [], [], [], [], [], [], [], [], []
    for i, row in tqdm(df.iterrows()):
        start = int(float(row["timesteps"].split(":")[0][1:]))
        end = int(float(row["timesteps"].split(":")[1][:-1]))
        starts.append(start)
        ends.append(end)
        id = count.get(row["vid_id"], 0)
        ids.append(row["vid_id"] + "_" + str(id))
        count[row["vid_id"]] = count.get(row["vid_id"], 0) + 1
        a1.append(row["a" + str(idx[i][0])])
        a2.append(row["a" + str(idx[i][1])])
        a3.append(row["a" + str(idx[i][2])])
        a4.append(row["a" + str(idx[i][3])])
        answer.append(idx[i].index(1))
        question.append(row["question"])
    return question, answer, ids, a1, a2, a3, a4, starts, ends

question, answer, ids, a1, a2, a3, a4, starts, ends = process(train_csv)
train_df = pd.DataFrame(
    {
        "question": question,
        "answer": answer,
        "video_id": ids,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a4": a4,
        "start": starts,
        "end": ends,
    },
    columns=["question", "answer", "video_id", "a1", "a2", "a3", "a4", "start", "end"],
)
print(len(train_df))

question, answer, ids, a1, a2, a3, a4, starts, ends = process(val_csv)
val_df = pd.DataFrame(
    {
        "question": question,
        "answer": answer,
        "video_id": ids,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a4": a4,
        "start": starts,
        "end": ends,
    },
    columns=["question", "answer", "video_id", "a1", "a2", "a3", "a4", "start", "end"],
)
print(len(val_df))

train_df.to_csv(os.path.join(HOW2QA_PATH, "train.csv"), index=False)
val_df.to_csv(os.path.join(HOW2QA_PATH, "val.csv"), index=False)
val_df.to_csv(os.path.join(HOW2QA_PATH, "test.csv"), index=False) # evaluation on the public val