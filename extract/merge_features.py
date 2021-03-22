import numpy as np
import argparse
import os
import torch
from tqdm import tqdm
import pandas as pd
from global_parameters import MSVD_PATH, TGIF_PATH, HOW2QA_PATH

parser = argparse.ArgumentParser(description="Feature merger")

parser.add_argument("--folder", type=str, required=True, help="folder of features")
parser.add_argument(
    "--output_path", type=str, required=True, help="output path for features"
)
parser.add_argument(
    "--dataset",
    type=str,
    help="dataset",
    required=True,
    choices=["ivqa", "msrvtt", "msvd", "tgif", "activitynet", "how2qa"],
)
parser.add_argument(
    "--pad",
    type=int,
    help="set as diff of 0 to trunc and pad up to a certain nb of seconds",
    default=0,
)

args = parser.parse_args()
files = os.listdir(args.folder)
files = [x for x in files if x[-4:] == ".npy"]

# Get mapping from feature file name to dataset video_id
if args.dataset == "msrvtt":
    mapping = {x: int(x.split(".")[0][5:]) for x in files}

elif args.dataset == "msvd":
    f = open(os.path.join(MSVD_PATH, "youtube_mapping.txt"))
    mapping = {}
    for line in f.readlines():
        l = line.split(" ")
        idx = l[1].split("\n")[0][3:]
        mapping[l[0] + ".avi.npy"] = int(idx)

elif args.dataset == "tgif":
    traindf = pd.read_csv(os.path.join(TGIF_PATH, "train.csv"))
    testdf = pd.read_csv(os.path.join(TGIF_PATH, "test.csv"))
    df = pd.concat([traindf, testdf])
    mapping = {}
    for x in tqdm(files):
        videodf = df[df["gif_name"] == x.split(".")[0]]["video_id"].values
        mapping[x] = videodf[0]
    files = list(mapping.keys())

elif args.dataset in ["ivqa", "activitynet"]:
    mapping = {x: x[:11] for x in files}

elif args.dataset == "how2qa":
    traindf = pd.read_csv(os.path.join(HOW2QA_PATH, "train.csv"))
    valdf = pd.read_csv(os.path.join(HOW2QA_PATH, "val.csv"))
    testdf = pd.read_csv(os.path.join(HOW2QA_PATH, "test.csv"))
    df = pd.concat([traindf, valdf, testdf])
    clip2timesteps = {}
    for i, x in df.iterrows():
        clip2timesteps[x['video_id']] = [x['start'], x['end']]

else:
    raise NotImplementedError

features = {}
if args.dataset != "how2qa":
    for i in tqdm(range(len(files))):
        x = files[i]
        feat = torch.from_numpy(np.load(os.path.join(args.folder, x)))
        if args.pad and len(feat) < args.pad:
            feat = torch.cat([feat, torch.zeros(args.pad - len(feat), feat.shape[1])])
        elif args.pad:
            feat = feat[: args.pad]
        features[mapping[x]] = feat.float()
else:
    for x in tqdm(clip2timesteps):
        video_id = x[:11]
        start, end = clip2timesteps[x]
        feat = torch.from_numpy(np.load(os.path.join(args.folder, video_id + '.mp4')))
        feat = feat[start: end + 1]
        if args.pad and len(feat) < args.pad:
            feat = torch.cat([feat, torch.zeros(args.pad - len(feat), feat.shape[1])])
        elif args.pad:
            feat = feat[: args.pad]
        features[mapping[x]] = feat.float()

torch.save(features, args.output_path)
