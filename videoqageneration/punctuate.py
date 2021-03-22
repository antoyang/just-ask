import pickle
import random
import os
from tqdm import tqdm
import string
import numpy as np
from punctuator import Punctuator
import sys

sys.path.insert(0, os.getcwd())
from global_parameters import PUNCTUATOR_PATH, HOWTO_PATH, punct_dir

# Load BRNN Punctuator Model
p = Punctuator(PUNCTUATOR_PATH)

# Load narrated videos from HowTo100M
videos = pickle.load(
    open(
        os.path.join(HOWTO_PATH, "caption_howto100m_sw_nointersec_norepeat.pickle"),
        "rb",
    )
)

# Shuffle video ids.
# This enables to parallelize the process by simply launching it on multiple machines
ids = list(videos.keys())
random.shuffle(ids)

# Iterate over all 1.2M unique videos
for vid in tqdm(ids):
    # If the video is already punctuated, skip
    if not os.path.exists(os.path.join(punct_dir, str(vid) + ".pkl")):
        # Punctuating the following video
        video = videos[vid]
        # Get cumulative lengths of the narration
        # This will enable to recover the ASR timesteps for the sentences
        lengths = np.cumsum([len(str(x)) + 1 for x in video["text"]])
        lengths[-1] -= 1

        # Apply Punctuator model
        punctuated_narration = p.punctuate(" ".join([str(x) for x in video["text"]]))
        # Split by sentences
        sentences = punctuated_narration.split(".")

        # Recover ASR timesteps of the sentences
        length = 0
        start_offset = 0
        end_offset = 0

        starts = []
        ends = []
        texts = []
        for sent in sentences:
            sen_len = len(
                str(sent).translate(str.maketrans("", "", string.punctuation))
            )
            if (
                len(str(sent).lstrip().rstrip().split(" ")) < 2
            ):  # Ignore sentences of 1 word
                length += sen_len
                continue

            texts.append(sent)

            # First start below len
            start_idx = start_offset + next(
                x[0] for x in enumerate(lengths[start_offset:]) if x[1] > length
            )
            start = video["start"][start_idx]
            starts.append(start)

            # First end above len + sen_len
            end_idx = end_offset + next(
                x[0]
                for x in enumerate(lengths[end_offset:])
                if x[1] >= length + sen_len
            )
            end = video["end"][end_idx]
            ends.append(end)

            # Update current length, start and end
            length += sen_len
            start_offset = start_idx
            end_offset = end_idx

        # Save
        if len(starts):
            punctuated_video = {"start": starts, "end": ends, "text": texts}
            pickle.dump(
                punctuated_video, open(os.path.join(punct_dir, str(vid) + ".pkl"), "wb")
            )
