import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from torch.utils.data.dataloader import default_collate
from util import tokenize


class WebVidVQA_Dataset(Dataset):
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
        feature_dim=1024,
    ):
        """
        Difference with HowToVQA dataloader: here all QAs from a video correspond to the whole video (no start and end timestamps)
        :param csv_path: path to a csv with video_id and video_path columns
        :param caption: dictionary mapping video_id to a dictionary mapping start, end, question and answer to corresponding lists
        :param features_path: path to the directory of features
        :param qmax_words: maximum number of words in the question
        :param amax_words: maximum number of words in the answer
        :param train: whether to train or validate
        :param n_pair: number of clips to sample from each video
        :param max_feats: maximum number of video features
        :param bert_tokenizer: BERT tokenizer
        :param feature_dim: dimension of the visual features
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
        self.feature_dim = feature_dim

    def __len__(self):
        return len(self.data)

    def _get_text(self, caption, n_pair_max, train=True):
        n_caption = len(caption["question"])
        n_pair_max = min(n_caption, n_pair_max)
        atxt = [""] * n_pair_max
        qtxt = [""] * n_pair_max
        r_ind = (
            np.random.choice(range(n_caption), n_pair_max, replace=False)
            if train
            else np.arange(n_pair_max)
        )  # sample clips

        for i in range(n_pair_max):
            ind = r_ind[i]
            if isinstance(caption["question"][ind], list):
                idx = (
                    np.random.randint(len(caption["question"][ind]))
                    if self.train
                    else 0
                )
                atxt[i], qtxt[i] = (
                    str(caption["answer"][ind]),
                    str(caption["question"][ind][idx]),
                )
            else:
                atxt[i], qtxt[i] = (
                    str(caption["answer"][ind]),
                    str(caption["question"][ind]),
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

        return atxt, answer, qtxt, question

    def _get_video(self, vid_path):
        feature_path = os.path.join(self.feature_path, vid_path)
        try:
            video = torch.from_numpy(np.load(feature_path)).float()
        except FileNotFoundError:
            video = (
                torch.zeros(1, self.feature_dim)
            )
        if len(video) > self.max_feats:
            video_len = self.max_feats
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            padded_video = torch.stack(sampled)
        else:
            padded_video = video
            video_len = len(video)
        feature = padded_video.unsqueeze(0)
        video_len = np.array([video_len], dtype=np.int)

        return feature, video_len

    def __getitem__(self, idx):
        video_id = str(self.data["video_id"].values[idx])
        if "feature_path" in self.data:
            video_path = self.data["feature_path"].values[idx]
        else:
            video_path = video_id + ".mp4.npy"
        atxt, answer, qtxt, question = self._get_text(
            self.caption[video_id], self.n_pair, train=self.train
        )
        video, video_len = self._get_video(video_path)

        return {
            "video_id": video_id,
            "video_path": video_path,
            "atxt": atxt,
            "qtxt": qtxt,
            "start": np.array(
                [0], dtype=np.int
            ),  # for compatibility with HowToVQA collate function
            "end": np.array(
                [0], dtype=np.int
            ),  # for compatibility with HowToVQA collate function
            "video": video,
            "video_len": video_len,
            "answer": answer,
            "question": question,
        }
