import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from torch.utils.data.dataloader import default_collate
from util import tokenize


class HowToVQA_Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        caption,
        features_path,
        qmax_words=20,
        amax_words=20,
        train=True,
        n_pair=32,
        max_feats=20,
        bert_tokenizer=None,
    ):
        """
        :param csv_path: path to a csv with video_id and video_path columns
        :param caption: dictionary mapping video_id to a dictionary mapping start, end, question and answer to corresponding lists
        :param features_path: path to the directory of features
        :param qmax_words: maximum number of words in the question
        :param amax_words: maximum number of words in the answer
        :param train: whether to train or validate
        :param n_pair: number of clips to sample from each video
        :param max_feats: maximum number of video features
        :param bert_tokenizer: BERT tokenizer
        """
        self.data = pd.read_csv(csv_path)
        self.caption = caption
        self.feature_path = features_path
        self.qmax_words = qmax_words
        self.amax_words = amax_words
        self.train = train
        self.n_pair = n_pair
        self.max_feats = max_feats
        self.bert_tokenizer = bert_tokenizer

    def __len__(self):
        return len(self.data)

    def _get_text(self, caption, n_pair_max, train=True):
        n_caption = len(caption["start"])
        n_pair_max = min(n_caption, n_pair_max)
        start = np.zeros(n_pair_max)
        end = np.zeros(n_pair_max)
        atxt = [""] * n_pair_max
        qtxt = [""] * n_pair_max
        r_ind = (
            np.random.choice(range(n_caption), n_pair_max, replace=False)
            if train
            else np.arange(n_pair_max)
        )  # sample clips

        for i in range(n_pair_max):
            ind = r_ind[i]
            atxt[i], qtxt[i], start[i], end[i] = (
                str(caption["answer"][ind]),
                str(caption["question"][ind]),
                caption["start"][ind],
                caption["end"][ind],
            )

        question = tokenize(
            qtxt,
            self.bert_tokenizer,
            add_special_tokens=True,
            max_length=self.qmax_words,
            dynamic_padding=True,
            truncation=True,
        )
        answer = tokenize(
            atxt,
            self.bert_tokenizer,
            add_special_tokens=True,
            max_length=self.amax_words,
            dynamic_padding=True,
            truncation=True,
        )

        return start, end, atxt, answer, qtxt, question

    def _get_video(self, vid_path, start, end):
        feature_path = os.path.join(self.feature_path, vid_path)
        video = torch.from_numpy(np.load(feature_path)).float()
        video_len = np.zeros(len(start))
        feature = torch.zeros(len(start), self.max_feats, video.shape[-1])

        for i in range(len(start)):
            s = int(start[i])
            e = int(end[i]) + 1
            slice = video[s:e]
            video_len[i] = min(self.max_feats, len(slice))
            if len(slice) < self.max_feats:
                padded_slice = torch.cat(
                    [slice, torch.zeros(self.max_feats - len(slice), slice.shape[1])]
                )
            else:
                padded_slice = slice[: self.max_feats]
            feature[i] = padded_slice

        return feature, video_len

    def __getitem__(self, idx):
        video_id = self.data["video_id"].values[idx]
        video_path = self.data["video_path"].values[idx]
        start, end, atxt, answer, qtxt, question = self._get_text(
            self.caption[video_id], self.n_pair, train=self.train
        )
        video, video_len = self._get_video(video_path, start, end)

        return {
            "video_id": video_id,
            "video_path": video_path,
            "atxt": atxt,
            "qtxt": qtxt,
            "start": start,
            "end": end,
            "video": video,
            "video_len": video_len,
            "answer": answer,
            "question": question,
        }


def howtovqa_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the question and the ans candidates padded to the max length of the batch
    """
    bs = len(batch)
    video_id = default_collate([batch[i]["video_id"] for i in range(bs)])
    video_path = default_collate([batch[i]["video_path"] for i in range(bs)])
    atxt = [batch[i]["atxt"] for i in range(bs)]
    atxt = [x for y in atxt for x in y]
    qtxt = [batch[i]["qtxt"] for i in range(bs)]
    qtxt = [x for y in qtxt for x in y]
    start = torch.cat([torch.from_numpy(batch[i]["start"]) for i in range(bs)], 0)
    end = torch.cat([torch.from_numpy(batch[i]["end"]) for i in range(bs)], 0)

    video = torch.cat([batch[i]["video"] for i in range(bs)], 0)
    video_len = torch.cat(
        [torch.from_numpy(batch[i]["video_len"]) for i in range(bs)], 0
    )

    ans = [batch[i]["answer"] for i in range(bs)]
    maxalen = max([x.shape[1] for x in ans])
    answer = torch.zeros(sum(x.shape[0] for x in ans), maxalen).long()
    idx = 0
    for i, tensor in enumerate(ans):
        n, l = tensor.shape
        answer[idx : idx + n, :l] = tensor
        idx += n

    que = [batch[i]["question"] for i in range(bs)]
    maxquelen = max([x.shape[1] for x in que])
    question = torch.zeros(sum(x.shape[0] for x in que), maxquelen).long()
    idx = 0
    for i, tensor in enumerate(que):
        n, l = tensor.shape
        question[idx : idx + n, :l] = tensor
        idx += n

    return {
        "video_id": video_id,
        "video_path": video_path,
        "atxt": atxt,
        "qtxt": qtxt,
        "start": start,
        "end": end,
        "video": video,
        "video_len": video_len,
        "answer": answer,
        "question": question,
    }
