import pickle
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import math
import torch
import sys

from global_parameters import answers_dir, QG_REPO_DIR, HOWTO_PATH, TRANSFORMERS_PATH

sys.path.insert(0, os.path.join(QG_REPO_DIR, "question_generation"))
from pipelines import pipeline


class Answer_Extraction_Dataset(Dataset):
    def __init__(self, caption, tokenizer):
        self.data = caption  # dictionnary mapping vid_id to lists of text sentences
        self.video_ids = list(caption.keys())
        self.tokenizer = tokenizer

    def _prepare_inputs_for_ans_extraction(self, sents):
        # prepare inputs for answer extraction
        inputs = []
        for sent in sents:
            sent = str(sent).strip()
            sent = " ".join(sent.split())
            source_text = "extract answers:"
            sent = "<hl> %s <hl>" % sent
            source_text = "%s %s" % (source_text, sent)
            source_text = source_text.strip()
            inputs.append(source_text)
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
        video_id = self.video_ids[index]
        text = self.data[video_id]["text"]
        inputs = self._prepare_inputs_for_ans_extraction(text)
        inputs = self._tokenize(inputs)

        return {
            "text": text,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "video_id": video_id,
        }

    def __len__(self):
        return len(self.data)


parser = argparse.ArgumentParser("")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--max_length", type=int, default=32)
args = parser.parse_args()

# Load infered sentences from speech transcripts
videos = pickle.load(
    open(
        os.path.join(
            HOWTO_PATH, "caption_howto100m_sw_nointersec_norepeat_punctuated.pickle"
        ),
        "rb",
    )
)

done = os.listdir(answers_dir)
doneset = set(x[:11] for x in done)
videos = {x: y for x, y in videos.items() if x not in doneset}

# Answer extraction transformer model
ans_tokenizer = AutoTokenizer.from_pretrained(
    "valhalla/t5-small-qa-qg-hl", cache_dir=TRANSFORMERS_PATH
)
ans_model = AutoModelForSeq2SeqLM.from_pretrained(
    "valhalla/t5-small-qa-qg-hl", cache_dir=TRANSFORMERS_PATH
)
ans_model.cuda()

# Dataloader - shuffle so that if this script can be parallelized on an arbitrary number of GPUs
dataset = Answer_Extraction_Dataset(caption=videos, tokenizer=ans_tokenizer)
dataloader = DataLoader(dataset, batch_size=1, num_workers=args.n_workers, shuffle=True)

# Inference
for i, batch in tqdm(enumerate(dataloader)):
    text, input_ids, attention_mask, video_id = (
        batch["text"],
        batch["input_ids"].squeeze(0).cuda(),
        batch["attention_mask"].squeeze(0).cuda(),
        batch["video_id"][0],
    )

    # Verify if the video has already been processed
    if os.path.exists(os.path.join(answers_dir, video_id + ".pkl")):
        continue

    # Batch inference
    n_iter = int(math.ceil(len(input_ids) / float(args.batch_size)))
    outs = torch.zeros(len(input_ids), args.max_length).long()
    with torch.no_grad():
        for k in range(n_iter):
            batch_outputs = (
                ans_model.generate(
                    input_ids=input_ids[
                        k * args.batch_size : (k + 1) * args.batch_size
                    ],
                    attention_mask=attention_mask[
                        k * args.batch_size : (k + 1) * args.batch_size
                    ],
                    max_length=args.max_length,
                )
                .detach()
                .cpu()
            )
            outs[
                k * args.batch_size : (k + 1) * args.batch_size, : batch_outputs.size(1)
            ] = batch_outputs

    # Decoding
    dec = [ans_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
    answers = [item.split("<sep>") for item in dec]
    answers = [i[:-1] for i in answers]
    answers = [
        list(set([y.strip() for y in x if len(y.strip())])) for x in answers
    ]  # remove duplicates
    answers = [
        [x for x in y if x in text[l] or x.capitalize() in text[l]]
        for l, y in enumerate(answers)
    ]  # remove answers that we cannot find back in the original sentence

    # Save
    if os.path.exists(os.path.join(answers_dir, video_id + ".pkl")):
        continue
    pickle.dump(answers, open(os.path.join(answers_dir, video_id + ".pkl"), "wb"))
