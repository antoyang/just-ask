import os
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from global_parameters import qas_dir, HOWTO_PATH

files = os.listdir(qas_dir)

sqa = {}
for file in tqdm(files):
    if os.path.exists(os.path.join(qas_dir, file)):
        try:
            video_qas = pickle.load(open(os.path.join(qas_dir, file), "rb"))
        except EOFError:
            continue
    video_id = file[:11]
    sqa[video_id] = video_qas

with open(os.path.join(HOWTO_PATH, "sqa.pkl"), "wb") as f:
    pickle.dump(sqa, f)

vids = set(sqa.keys())
howto_csv = pd.read_csv(os.path.join(HOWTO_PATH, "s3d_features_nointersec.csv"))
sqa_csv = howto_csv[howto_csv["video_id"].isin(vids)]
train_sqa, val_sqa = train_test_split(sqa_csv, test_size=0.1, random_state=0)
train_sqa.to_csv(os.path.join(HOWTO_PATH, "train_sqa.csv"), index=False)
val_sqa.to_csv(os.path.join(HOWTO_PATH, "val_sqa.csv"), index=False)
