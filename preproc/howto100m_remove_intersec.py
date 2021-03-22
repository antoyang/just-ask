import pickle
import pandas as pd
import json
import os

from global_parameters import IVQA_PATH, MSRVTT_PATH, MSVD_PATH, ACT_PATH, HOWTO_PATH, HOW2QA_PATH

# Load HowTo100M narrations and csv
videos = pickle.load(
    open(os.path.join(HOWTO_PATH, "caption_howto100m_with_stopwords.pkl"), "rb")
)
ids = set(videos.keys())
howto100m_csv = pd.read_csv(os.path.join(HOWTO_PATH, "s3d_features.csv"))

# Intersection with IVQA (all)
ivqa_train_csv = pd.read_csv(os.path.join(IVQA_PATH, "train.csv"))
ivqa_val_csv = pd.read_csv(os.path.join(IVQA_PATH, "val.csv"))
ivqa_test_csv = pd.read_csv(os.path.join(IVQA_PATH, "test.csv"))
ivqa_ids = (
    [x for i, x in enumerate(ivqa_train_csv["video_id"].values)]
    + [x for i, x in enumerate(ivqa_val_csv["video_id"].values)]
    + [x for i, x in enumerate(ivqa_test_csv["video_id"].values)]
)
intersec = [x for x in ivqa_ids if x in ids]

# Intersection with MSR-VTT (val and test)
trainval_msrvtt = json.load(
    open(
        os.path.join(MSRVTT_PATH, "train_val_annotation/train_val_videodatainfo.json"),
        "r",
    )
)["videos"]
# The following ids corresponds to the MSRVTT-QA val set
msrvtt_val_ids = [
    x["url"][-11:] for x in trainval_msrvtt if int(x["video_id"][5:]) > 6512
]
test_msrvtt = json.load(
    open(os.path.join(MSRVTT_PATH, "test_videodatainfo.json"), "r")
)["videos"]
msrvtt_test_ids = [x["url"][-11:] for x in test_msrvtt if int(x["video_id"][5:])]
msrvtt_ids = msrvtt_val_ids + msrvtt_test_ids
intersec += [x for x in msrvtt_ids if x in ids]

# Intersection with MSVD (val and test)
f = open(os.path.join(MSVD_PATH, "youtube_mapping.txt"))
msvd_ids = [("_".join(l.split(" ")[0].split("_")[:-2])) for l in f.readlines()]
# These ids corresponds to the MSVD-QA val and test set
intersec += [x for i, x in enumerate(msvd_ids) if x in ids and i > 1200]

# Intersection with ActivityNet (val and test)
activitynet_val = json.load(open(os.path.join(ACT_PATH, "dataset/val_q.json"), "rb"))
activitynet_test = json.load(open(os.path.join(ACT_PATH, "dataset/test_q.json"), "rb"))
activitynet_ids = set(
    [x["video_name"] for x in activitynet_test]
    + [x["video_name"] for x in activitynet_val]
)
intersec += [x for x in activitynet_ids if x in ids]

# Intersection with How2QA (public val)
how2qa_val = pd.read_csv(os.path.join(HOW2QA_PATH, "how2QA_val_release.csv"))
how2qa_ids = set(how2qa_val["vid_id"].values)
intersec += [x for x in how2qa_ids if x in ids]

intersec = set(intersec)
print(len(intersec))
howto100m_csv_nointersec = howto100m_csv[~howto100m_csv["video_id"].isin(intersec)]
howto100m_csv_nointersec.to_csv(
    os.path.join(HOWTO_PATH, "s3d_features_nointersec.csv"), index=False
)

videos_nointersec = {x: videos[x] for x in videos if x not in intersec}
pickle.dump(
    videos_nointersec,
    open(os.path.join(HOWTO_PATH, "caption_howto100m_sw_nointersec.pickle"), "wb"),
)
