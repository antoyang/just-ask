import os
import pickle
from tqdm import tqdm
import pandas as pd

from global_parameters import qas_dir, WEBVID_PATH

files = os.listdir(qas_dir)

# Merge all pickles into one file
sqa = {}
for file in tqdm(files):
    if os.path.exists(os.path.join(qas_dir, file)):
        try:
            video_qas = pickle.load(open(os.path.join(qas_dir, file), "rb"))
        except EOFError:
            continue
    video_id = file[:-4]
    # Remove qa pairs for which the answer is fully in the question
    idx = [
        i
        for i in range(len(video_qas["question"]))
        if video_qas["answer"][i] not in video_qas["question"][i]
    ]
    if not idx:
        continue
    video_qas["question"] = [video_qas["question"][i] for i in idx]
    video_qas["answer"] = [video_qas["answer"][i] for i in idx]
    sqa[video_id] = video_qas

print(len(sqa))
print(len([x for x in sqa for k in sqa[x]["question"]]))
with open(os.path.join(WEBVID_PATH, "webvidvqa.pkl"), "wb") as f:
    pickle.dump(sqa, f)

# Create csv
train = pd.read_csv(os.path.join(WEBVID_PATH, "results_2M_train.csv"))
val = pd.read_csv(os.path.join(WEBVID_PATH, "results_2M_val.csv"))
webvid_csv = pd.DataFrame(
    {"feature_path": [str(x) + '.mp4.npy' for x in sqa], "video_id": [str(x) for x in sqa]},
    columns=["feature_path", "video_id"],
)
train_csv = webvid_csv[webvid_csv["video_id"].isin(set(train["video_id"]))]
val_csv = webvid_csv[webvid_csv["video_id"].isin(set(val["video_id"]))]
train_csv.to_csv(os.path.join(WEBVID_PATH, "train_webvidvqa.csv"), index=False)
val_csv.to_csv(os.path.join(WEBVID_PATH, "val_webvidvqa.csv"), index=False)
