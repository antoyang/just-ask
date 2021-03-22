import os
import pickle
from tqdm import tqdm
import sys
sys.path.insert(0, os.getcwd())
from global_parameters import answers_dir, HOWTO_PATH

files = os.listdir(answers_dir)

videos = pickle.load(
    open(
        os.path.join(
            HOWTO_PATH, "caption_howto100m_sw_nointersec_norepeat_punctuated.pickle"
        ),
        "rb",
    )
)

sqa = {}
for file in tqdm(files):
    if os.path.exists(os.path.join(answers_dir, file)):
    # Load extracted answers
        try:
            video_answers = pickle.load(open(os.path.join(answers_dir, file), "rb"))
        except EOFError:
            continue

    video_id = file[:11]
    # Save as list of answers together with index mapping to the original text sentence
    answers_txt = []
    answers_idx = []
    for i, ans in enumerate(video_answers):
        ans = [y for y in ans if y in videos[video_id]["text"][i]]
        if ans:
            answers_txt.extend(ans)
            answers_idx.extend([i] * len(ans))

    if answers_txt:
        sqa[video_id] = {"answer": answers_txt, "idx": answers_idx}

with open(os.path.join(HOWTO_PATH, "sqa_answers.pickle"), "wb") as f:
    pickle.dump(sqa, f)
