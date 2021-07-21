import os
import pickle
from tqdm import tqdm

from global_parameters import answers_dir, HOWTO_PATH, HOWTOVQA_PATH

files = os.listdir(answers_dir)

videos = pickle.load(
    open(
        os.path.join(
            HOWTO_PATH, "caption_howto100m_sw_nointersec_norepeat_punctuated.pickle"
        ),
        "rb",
    )
)

howtovqa = {}
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
        howtovqa[video_id] = {"answer": answers_txt, "idx": answers_idx}

with open(os.path.join(HOWTOVQA_PATH, "howtovqa_answers.pickle"), "wb") as f:
    pickle.dump(howtovqa, f)
