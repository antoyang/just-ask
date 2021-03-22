import torch as th
from torch.utils.data import Dataset
import pandas as pd
import pickle


class VideoText_Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        features_path,
        max_words=30,
        bert_tokenizer=None,
        max_feats=20,
    ):
        """
        Args:
        """
        self.data = pd.read_csv(csv_path)
        self.features = th.load(features_path)
        self.max_words = max_words
        self.bert_tokenizer = bert_tokenizer
        self.max_feats = max_feats

    def __len__(self):
        return len(self.data)

    def _bert_tokenizer(self, text):
        tokens = th.tensor(
            self.bert_tokenizer.encode(
                text,
                add_special_tokens=True,
                padding="max_length",
                max_length=self.max_words,
                truncation=True,
            ),
            dtype=th.long,
        )
        return tokens

    def __getitem__(self, idx):
        text = self.data["sentence"].values[idx]
        text_embd = th.tensor(
            self.bert_tokenizer.encode(
                text,
                add_special_tokens=True,
                padding="longest",
                max_length=self.max_words,
                truncation=True,
            ),
            dtype=th.long,
        )

        video_id = self.data["video_id"].values[idx]
        video = self.features[video_id]
        if len(video) < self.max_feats:
            video = video[: self.max_feats]
            vid_duration = len(video)
            if len(video) < self.max_feats:  # pad
                video = th.cat(
                    [video, th.zeros(self.max_feats - len(video), video.shape[1])]
                )
        else:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = th.stack(sampled)
            vid_duration = len(video)
        return {"video": video, "video_len": vid_duration, "text": text_embd}


class Youcook_Dataset(Dataset):
    """Youcook dataset loader."""

    def __init__(
        self,
        data,
        max_words=30,
        bert_tokenizer=None,
        max_feats=20,
    ):
        """
        Args:
        """
        self.data = pickle.load(open(data, "rb"))
        self.max_words = max_words
        self.bert_tokenizer = bert_tokenizer
        self.max_feats = max_feats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["caption"]
        text_embd = th.tensor(
            self.bert_tokenizer.encode(
                text,
                add_special_tokens=True,
                padding="longest",
                max_length=self.max_words,
                truncation=True,
            ),
            dtype=th.long,
        )

        video = th.from_numpy(self.data[idx]["feature"]).float()
        if len(video) <= self.max_feats:
            video = video[: self.max_feats]
            vid_duration = len(video)
            if len(video) < self.max_feats:  # pad
                video = th.cat(
                    [video, th.zeros(self.max_feats - len(video), video.shape[1])]
                )
        else:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = th.stack(sampled)
            vid_duration = len(video)

        return {"video": video, "video_len": vid_duration, "text": text_embd}


def videotext_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the text padded to the max length of the batch
    """
    bs = len(batch)
    video = th.stack([batch[i]["video"] for i in range(bs)], 0)
    video_len = th.tensor([batch[i]["video_len"] for i in range(bs)], dtype=th.long)
    text = [batch[i]["text"] for i in range(bs)]
    maxlen = max([len(x) for x in text])
    text_padded = th.zeros(bs, maxlen).long()
    for i, tensor in enumerate(text):
        l = len(tensor)
        text_padded[i, :l] = tensor

    return {
        "video": video,
        "video_len": video_len,
        "text": text_padded,
    }
