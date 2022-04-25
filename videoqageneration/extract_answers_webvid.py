import pickle
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import torch
import sys
import pandas as pd

sys.path.insert(0, os.getcwd())  # to correct with parent folder
from global_parameters import answers_dir, QG_REPO_DIR, WEBVID_PATH, TRANSFORMERS_PATH

sys.path.insert(0, os.path.join(QG_REPO_DIR, "question_generation"))
from pipelines import pipeline


class Answer_Extraction_Dataset(Dataset):
    def __init__(self, caption, tokenizer):
        self.data = caption  # dictionnary mapping vid_id to text
        self.video_ids = list(caption.keys())
        self.tokenizer = tokenizer

    def _prepare_inputs_for_ans_extraction(self, sent):
        # prepare inputs for answer extraction
        sent = str(sent).strip()
        sent = " ".join(sent.split())
        source_text = "extract answers:"
        sent = "<hl> %s <hl>" % sent
        source_text = "%s %s" % (source_text, sent)
        source_text = source_text.strip()
        return source_text

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
        text = self.data[video_id]
        source_text = self._prepare_inputs_for_ans_extraction(text)
        inputs = self._tokenize([source_text])

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

# Load captions
train_videos = pd.read_csv(os.path.join(WEBVID_PATH, "results_10M_train.csv"))
val_videos = pd.read_csv(os.path.join(WEBVID_PATH, "results_2M_val.csv"))
print("Descriptions loaded")
videos = {}
for _, row in train_videos.iterrows():
    videos[row["videoid"]] = row["name"]
for _, row in val_videos.iterrows():
    videos[row["videoid"]] = row["name"]
done = os.listdir(answers_dir)
doneset = set(x[:-4] for x in done)
videos = {x: y for x, y in videos.items() if str(x) not in doneset}
print("Descriptions prepared")
print(len(videos))

# Answer extraction transformer model
ans_tokenizer = AutoTokenizer.from_pretrained(
    "valhalla/t5-small-qa-qg-hl", cache_dir=TRANSFORMERS_PATH
)
ans_model = AutoModelForSeq2SeqLM.from_pretrained(
    "valhalla/t5-small-qa-qg-hl", cache_dir=TRANSFORMERS_PATH
)
ans_model.cuda()
print("Models loaded")

# Dataloader - shuffle so that if this script can be parallelized on an arbitrary number of GPUs
dataset = Answer_Extraction_Dataset(caption=videos, tokenizer=ans_tokenizer)
dataloader = DataLoader(
    dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True
)
print("Dataloaders prepared")

# Inference
for i, batch in tqdm(enumerate(dataloader)):
    text, input_ids, attention_mask, video_id = (
        batch["text"],
        batch["input_ids"].squeeze(1).cuda(),
        batch["attention_mask"].squeeze(1).cuda(),
        batch["video_id"],
    )

    # Verify if the video has already been processed
    todo_batch_list = [
        j
        for j in range(len(video_id))
        if not os.path.exists(
            os.path.join(answers_dir, str(video_id[j].item()) + ".pkl")
        )
    ]
    if not len(todo_batch_list):
        continue
    text = [text[j] for j in todo_batch_list]
    todo_batch = torch.Tensor(todo_batch_list).long().cuda()
    input_ids = torch.index_select(input_ids, 0, todo_batch)
    attention_mask = torch.index_select(attention_mask, 0, todo_batch)

    # Batch inference
    with torch.no_grad():
        outs = (
            ans_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.max_length,
            )
            .detach()
            .cpu()
        )

    # Decoding
    dec = [ans_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
    answers = [item.split("<sep>") for item in dec]
    answers = [i[:-1] for i in answers]
    answers = [
        list(set([y.strip() for y in x if len(y.strip())])) for x in answers
    ]  # remove duplicates

    # Save
    for j, idx in enumerate(todo_batch_list):
        if not os.path.exists(
            os.path.join(answers_dir, str(video_id[idx].item()) + ".pkl")
        ):
            pickle.dump(
                answers[j],
                open(
                    os.path.join(answers_dir, str(video_id[idx].item()) + ".pkl"), "wb"
                ),
            )
