from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import torch
from global_parameters import HOW2QA_PATH, HOWTO_FEATURES_PATH

train_csv = pd.read_csv(os.path.join(HOW2QA_PATH, "how2QA_train_release.csv"))
train_csv.columns = ["vid_id", "timesteps", "a2", "a3", "a4", "question", "a1"]
print(len(train_csv)) # 35404
val_csv = pd.read_csv(os.path.join(HOW2QA_PATH, "how2QA_val_release.csv"))
val_csv.columns = ["vid_id", "timesteps", "a2", "a3", "a4", "question", "a1"]
print(len(val_csv)) # 2851

count = {}
missing_videos = []
missing_features = []
features = {}


def process(df):
    idx = [[1, 2, 3, 4] for _ in range(len(df))]
    for i in range(len(idx)):
        np.random.shuffle(idx[i])
    ids, a1, a2, a3, a4, answer, question, starts, ends = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, row in tqdm(df.iterrows()):
        feat_path = HOWTO_FEATURES_PATH + row["vid_id"] + ".mp4" + ".npy"
        if not os.path.exists(feat_path):
            feat_path = HOWTO_FEATURES_PATH + row["vid_id"] + ".webm" + ".npy"
        if not os.path.exists(feat_path):
            missing_videos.append(row["vid_id"])
            continue
        start = int(float(row["timesteps"].split(":")[0][1:]))
        end = int(float(row["timesteps"].split(":")[1][:-1]))
        feature = torch.from_numpy(np.load(feat_path))
        feature = feature[start : end + 1]
        if len(feature) != end - start + 1:
            missing_features.append((row["video_id"], start, end))
            continue
        starts.append(start)
        ends.append(end)
        id = count.get(row["vid_id"], 0)
        ids.append(row["vid_id"] + "_" + str(id))
        features[row["vid_id"] + "_" + str(id)] = feature.float()
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
print(len(train_df)) # 35070, about 0.9% missing videos or features

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
print(len(val_df)) # 2833, about 0.6% missing videos or features

print(missing_videos) # 8e_jI7rLB04
print(len(missing_features)) # 239

train_df.to_csv(os.path.join(HOW2QA_PATH, "train.csv"), index=False)
val_df.to_csv(os.path.join(HOW2QA_PATH, "val.csv"), index=False)
val_df.to_csv(
    os.path.join(HOW2QA_PATH, "test.csv"), index=False
)  # evaluation on the public val
torch.save(features, os.path.join(HOW2QA_PATH, "s3d.pth"))