import pickle
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
import math
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

from global_parameters import TRANSFORMERS_PATH, qas_dir, HOWTO_PATH


class Question_Generation_Dataset(Dataset):
    def __init__(self, caption, ext_answers, tokenizer):
        self.data = caption  # dictionary mapping vid_id to lists of text, start, end
        self.answers = ext_answers  # dictionary mapping vid_id to list of answers, index corresponding to the original text
        self.video_ids = list(ext_answers.keys())
        self.tokenizer = tokenizer

    def _prepare_inputs_for_qg_from_answers_hl(self, text, answers):
        # prepare inputs for answer-aware question generation
        inputs = []
        for (sent, a) in zip(text, answers):
            try:
                start = sent.index(a)
            except ValueError: # substring not found
                start = text.index(a.capitalize())
            text_hl = f"{sent[:start]} <hl> {a} <hl> {sent[start + len(a):]}"
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
        text = self.data[vid]["text"]
        answer = self.answers[vid]["answer"]
        indices = self.answers[vid]["idx"]
        text = [text[i] for i in indices]
        qg_inputs = self._prepare_inputs_for_qg_from_answers_hl(text, answer)
        inputs = self._tokenize(qg_inputs, padding=True, truncation=True)
        start = torch.tensor([self.data[vid]["start"][i] for i in indices])
        end = torch.tensor([self.data[vid]["end"][i] for i in indices])

        return {
            "text": text,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "answers": answer,
            "video_id": vid,
            "start": start,
            "end": end,
        }

    def __len__(self):
        return len(self.video_ids)


parser = argparse.ArgumentParser("")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--max_length", type=int, default=32)
parser.add_argument("--num_beams", type=int, default=4)
args = parser.parse_args()

# Load speech transcripts and extracted answers
videos = pickle.load(
    open(
        os.path.join(
            HOWTO_PATH, "caption_howto100m_sw_nointersec_norepeat_punctuated.pickle"
        ),
        "rb",
    )
)
ext_answers = pickle.load(
    open(
        os.path.join(HOWTO_PATH, "howtovqa_answers.pickle"),
        "rb",
    )
)

done = os.listdir(qas_dir)
doneset = set(x[:11] for x in done)
videos = {x: y for x, y in videos.items() if x not in doneset}
ext_answers = {x: y for x, y in ext_answers.items() if x not in doneset}

# Answer-aware question generation transformer model
tokenizer = AutoTokenizer.from_pretrained(
    "valhalla/t5-base-qg-hl", cache_dir=TRANSFORMERS_PATH
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "valhalla/t5-base-qg-hl", cache_dir=TRANSFORMERS_PATH
)
model.cuda()

# Dataloader
dataset = Question_Generation_Dataset(
    caption=videos, ext_answers=ext_answers, tokenizer=tokenizer
)
dataloader = DataLoader(dataset, batch_size=1, num_workers=args.n_workers, shuffle=True)
for i, batch in tqdm(enumerate(dataloader)):
    text, input_ids, attention_mask, answers, video_id, start, end = (
        batch["text"],
        batch["input_ids"].squeeze(0).cuda(),
        batch["attention_mask"].squeeze(0).cuda(),
        batch["answers"],
        batch["video_id"][0],
        batch["start"].squeeze(0),
        batch["end"].squeeze(0),
    )

    # Verify if the video has already been processed
    if os.path.exists(os.path.join(qas_dir, video_id + ".pkl")):
        continue

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
    answers = [answers[l][0] for l in range(len(answers))]
    text = [text[l][0] for l in range(len(answers))]

    # Remove qa pairs for which the answer is fully in the question
    idx = [k for k in range(len(questions)) if answers[k] not in questions[k]]
    questions = [questions[k] for k in idx]
    answers = [answers[k] for k in idx]
    text = [text[k] for k in idx]
    start = [start[k].item() for k in idx]
    end = [end[k].item() for k in idx]

    # Save
    if os.path.exists(os.path.join(qas_dir, video_id + ".pkl")):
        continue
    pickle.dump(
        {
            "text": text,
            "question": questions,
            "answer": answers,
            "start": start,
            "end": end,
        },
        open(os.path.join(qas_dir, video_id + ".pkl"), "wb"),
    )
