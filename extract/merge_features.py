import numpy as np
import argparse
import os
import torch
from tqdm import tqdm
import pandas as pd
from global_parameters import MSVD_PATH, HOW2QA_PATH

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
    choices=["ivqa", "msrvtt", "msvd", "activitynet"],
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

elif args.dataset in ["ivqa", "activitynet"]:
    mapping = {x: x[:11] for x in files}

else:
    raise NotImplementedError

features = {}
for i in tqdm(range(len(files))):
    x = files[i]
    feat = torch.from_numpy(np.load(os.path.join(args.folder, x)))
    if args.pad and len(feat) < args.pad:
        feat = torch.cat([feat, torch.zeros(args.pad - len(feat), feat.shape[1])])
    elif args.pad:
        feat = feat[: args.pad]
    features[mapping[x]] = feat.float()

torch.save(features, args.output_path)
