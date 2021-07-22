import os
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from global_parameters import qas_dir, HOWTO_PATH, HOWTOVQA_PATH

files = os.listdir(qas_dir)

howtovqa = {}
for file in tqdm(files):
    if os.path.exists(os.path.join(qas_dir, file)):
        try:
            video_qas = pickle.load(open(os.path.join(qas_dir, file), "rb"))
        except EOFError:
            continue
    video_id = file[:11]
    howtovqa[video_id] = video_qas

with open(os.path.join(HOWTOVQA_PATH, "howtovqa.pkl"), "wb") as f:
    pickle.dump(howtovqa, f)

vids = set(howtovqa.keys())
howto_csv = pd.read_csv(os.path.join(HOWTO_PATH, "s3d_features_nointersec.csv"))
howtovqa_csv = howto_csv[howto_csv["video_id"].isin(vids)]
train_howtovqa, val_howtovqa = train_test_split(
    howtovqa_csv, test_size=0.1, random_state=0
)
train_howtovqa.to_csv(os.path.join(HOWTOVQA_PATH, "train_howtovqa.csv"), index=False)
val_howtovqa.to_csv(os.path.join(HOWTOVQA_PATH, "val_howtovqa.csv"), index=False)
