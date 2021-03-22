import os
import pickle
from tqdm import tqdm

# FILL
from global_parameters import punct_dir, HOWTO_PATH

files = os.listdir(punct_dir)

punctuated_videos = {}
for file in tqdm(files):
    if os.path.exists(os.path.join(punct_dir, file)):
        try:
            pic = pickle.load(open(os.path.join(punct_dir, file), "rb"))
        except EOFError:
            continue
    else:
        continue
    video_id = file[:11]
    if "text" in pic and len(pic["text"]):
        punctuated_videos[video_id] = pic

with open(
    os.path.join(
        HOWTO_PATH, "caption_howto100m_sw_nointersec_norepeat_punctuated.pickle"
    ),
    "wb",
) as f:
    pickle.dump(punctuated_videos, f)
