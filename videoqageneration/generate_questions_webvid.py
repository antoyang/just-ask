import pickle
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
import math
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import sys
import pandas as pd

sys.path.insert(0, os.getcwd())
from global_parameters import TRANSFORMERS_PATH, qas_dir, WEBVID_PATH


class Question_Generation_Dataset(Dataset):
    def __init__(self, caption, ext_answers, tokenizer):
        self.data = caption  # dictionary mapping vid_id to text
        self.answers = ext_answers  # dictionary mapping vid_id to list of answers
        self.video_ids = list(ext_answers.keys())
        self.tokenizer = tokenizer

    def _prepare_inputs_for_qg_from_answers_hl(self, text, answers):
        # prepare inputs for answer-aware question generation
        inputs = []
        for a in answers:
            try:
                start = text.index(a)
            except ValueError:  # substring not found
                start = text.index(
                    a.capitalize()
                )  # preremoved the 2% examples that leads to substring not found anyway
            text_hl = f"{text[:start]} <hl> {a} <hl> {text[start + len(a):]}"
            input = f"generate question: {text_hl}"
            inputs.append(input)
        return inputs

    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512,
    ):
        # batch tokenizer
        inputs = self.tokenizer(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            return_tensors="pt",
        )
        return inputs

    def __getitem__(self, index):
        vid = self.video_ids[index]
        text = self.data[vid]
        answer = self.answers[vid]
        qg_inputs = self._prepare_inputs_for_qg_from_answers_hl(text, answer)
        inputs = self._tokenize(qg_inputs, padding=True, truncation=True)

        return {
            "text": text,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "answers": answer,
            "video_id": vid,
        }

    def __len__(self):
        return len(self.video_ids)


def qgen_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the question and the ans candidates padded to the max length of the batch
    """
    text = [x["text"] for x in batch]
    input_ids = torch.cat([x["input_ids"] for x in batch], 0)
    attention_mask = torch.cat([x["attention_mask"] for x in batch], 0)
    answers = [x["answers"] for x in batch]
    video_id = [x["video_id"] for x in batch]

    return {
        "text": text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answers": answers,
        "video_id": video_id,
    }


parser = argparse.ArgumentParser("")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--max_length", type=int, default=32)
parser.add_argument("--num_beams", type=int, default=4)
args = parser.parse_args()

# Load captions
train_videos = pd.read_csv(os.path.join(WEBVID_PATH, "results_2M_train.csv"))
val_videos = pd.read_csv(os.path.join(WEBVID_PATH, "results_2M_val.csv"))
print("Descriptions loaded")
videos = {}
for _, row in train_videos.iterrows():
    videos[row["videoid"]] = row["name"]
for _, row in val_videos.iterrows():
    videos[row["videoid"]] = row["name"]
done = os.listdir(qas_dir)
doneset = set(x[:-4] for x in done)
videos = {x: y for x, y in videos.items() if str(x) not in doneset}
print("Descriptions prepared")
print(len(videos))

# Load answers
ext_answers = pickle.load(
    open(os.path.join(WEBVID_PATH, "webvid_answers.pkl"), "rb")
)
ext_answers = {int(x): y for x, y in ext_answers.items() if str(x) not in doneset}
videos = {x: y for x, y in videos.items() if x in ext_answers}
print(len(videos))
print(len(ext_answers))

# Answer-aware question generation transformer model
tokenizer = AutoTokenizer.from_pretrained(
    "valhalla/t5-base-qg-hl", cache_dir=TRANSFORMERS_PATH
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "valhalla/t5-base-qg-hl", cache_dir=TRANSFORMERS_PATH
)
model.cuda()
print("Models loaded")

# Dataloader
dataset = Question_Generation_Dataset(
    caption=videos, ext_answers=ext_answers, tokenizer=tokenizer
)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.n_workers,
    shuffle=True,
    collate_fn=qgen_collate_fn,
)
print("Dataloaders prepared")

for i, batch in tqdm(enumerate(dataloader)):
    text, input_ids, attention_mask, answers, video_id = (
        batch["text"],
        batch["input_ids"].squeeze(1).cuda(),
        batch["attention_mask"].squeeze(1).cuda(),
        batch["answers"],
        batch["video_id"],
    )

    # Verify if the video has already been processed
    todo_batch_list = [
        j
        for j in range(len(video_id))
        if not os.path.exists(os.path.join(qas_dir, str(video_id[j]) + ".pkl"))
    ]
    if not len(todo_batch_list):
        continue
    text = [text[j] for j in todo_batch_list]
    todo_batch_list_a = []
    idx = 0
    todo_batch_set = set(todo_batch_list)
    for j, ans in enumerate(answers):
        if j in todo_batch_list:
            todo_batch_list_a.extend([idx + k for k in range(len(answers[j]))])
        idx += len(answers[j])
    todo_batch = torch.Tensor(todo_batch_list_a).long().cuda()
    input_ids = torch.index_select(input_ids, 0, todo_batch)
    attention_mask = torch.index_select(attention_mask, 0, todo_batch)
    answers = [answers[j] for j in todo_batch_list]

    # Batch inference
    n_iter = int(math.ceil(len(input_ids) / float(args.batch_size)))
    outs = torch.zeros(len(input_ids), args.max_length).long()
    with torch.no_grad():
        for k in range(n_iter):
            batch_outputs = (
                model.generate(
                    input_ids=input_ids[
                        k * args.batch_size : (k + 1) * args.batch_size
                    ],
                    attention_mask=attention_mask[
                        k * args.batch_size : (k + 1) * args.batch_size
                    ],
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                )
                .detach()
                .cpu()
            )
            outs[
                k * args.batch_size : (k + 1) * args.batch_size, : batch_outputs.size(1)
            ] = batch_outputs

    # Decoding
    questions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

    # Save
    video_id = [video_id[j] for j in todo_batch_list]
    qidx = 0
    for j, vid in enumerate(video_id):
        if not os.path.exists(os.path.join(qas_dir, str(vid) + ".pkl")):
            pickle.dump(
                {
                    "text": text[j],
                    "question": questions[qidx : qidx + len(answers[j])],
                    "answer": answers[j],
                },
                open(os.path.join(qas_dir, str(vid) + ".pkl"), "wb"),
            )
        qidx += len(answers[j])