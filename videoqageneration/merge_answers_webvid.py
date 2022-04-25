import os
import pickle
from tqdm import tqdm
import sys
import pandas as pd

sys.path.insert(0, os.getcwd())
from global_parameters import answers_dir, WEBVID_PATH

files = os.listdir(answers_dir)

# Map video id to caption
train = pd.read_csv(os.path.join(WEBVID_PATH, "results_2M_train.csv"))
val = pd.read_csv(os.path.join(WEBVID_PATH, "results_2M_val.csv"))
df = pd.concat([train, val])
vid2text = {x["videoid"]: x["name"] for _, x in df.iterrows()}

sqa = {}
for file in tqdm(files):
    if os.path.exists(os.path.join(answers_dir, file)):
        # Load extracted answers
        try:
            video_answers = pickle.load(open(os.path.join(answers_dir, file), "rb"))
        except EOFError:
            continue

    video_id = file[:-4]
    # Save as list of answers
    video_answers = [y for y in video_answers if y in vid2text[video_id] or y.capitalize() in vid2text[video_id]]

    if len(video_answers):
        sqa[video_id] = video_answers

print(len(sqa))
with open(os.path.join(WEBVID_PATH, "webvid_answers.pkl"), "wb") as f:
    pickle.dump(sqa, f)
