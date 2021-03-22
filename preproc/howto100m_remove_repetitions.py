import pickle
from typing import List
from tqdm import tqdm
import os

from global_parameters import HOWTO_PATH


def remove_rep(
    clips_text: List[str], clips_start: List[int], clips_end: List[int]
) -> List[str]:
    # Used to remove repetitions in the ASR of a narrated video
    texts, starts, ends = [], [], []
    texts.append(clips_text[0])
    starts.append(clips_start[0])
    ends.append(clips_end[0])
    for i in range(1, len(clips_text)):
        words_prev = str(texts[-1]).split(" ")
        words_next = str(clips_text[i]).split(" ")
        removed_rep = False
        max_subset_size = min(len(words_prev), len(words_next))
        for s in range(max_subset_size, 1, -1):
            if words_prev[-s:] == words_next[:s]:
                # If the captions are literally the same, just make a bigger clip.
                if s == len(words_prev) and len(words_prev) == len(words_next):
                    ends[-1] = clips_end[i]
                else:
                    # Remove repeated words from the largest caption between the two.
                    if len(words_next) >= len(words_prev):
                        texts.append(" ".join(words_next[s:]))
                    else:
                        texts[-1] = " ".join(words_prev[:-s])
                        texts.append(" ".join(words_next))
                    starts.append(clips_start[i])
                    ends.append(clips_end[i])
                removed_rep = True
                break
        if not removed_rep:
            texts.append(" ".join(words_next))
            starts.append(clips_start[i])
            ends.append(clips_end[i])
    return texts, starts, ends


videos = pickle.load(
    open(os.path.join(HOWTO_PATH, "caption_howto100m_sw_nointersec.pkl"), "rb")
)

videos_norep = {}
for vid in tqdm(videos):
    video = videos[vid]
    if video["text"]:
        video_norep = {}
        video_norep["text"], video_norep["start"], video_norep["end"] = remove_rep(
            video["text"], video["start"], video["end"]
        )
        videos_norep[vid] = video_norep

pickle.dump(
    videos_norep,
    open(
        os.path.join(HOWTO_PATH, "caption_howto100m_sw_nointersec_norepeat.pickle"),
        "wb",
    ),
)
