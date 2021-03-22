import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from util import tokenize


class HowTo_Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        caption,
        features_path,
        min_time=10,
        max_time=20,
        min_words=10,
        max_words=20,
        n_pair=32,
        bert_tokenizer=None,
    ):
        """
        :param csv: path to a csv with video_id and video_path columns
        :param caption: caption: dictionary mapping video_id to a dictionary mapping start, end, text to corresponding lists
        :param features_path: path to the directory of features
        :param min_time: minimum duration of a clip in seconds
        :param max_time: maximum duration of a clip in seconds
        :param min_words: minimum number of words in a clip
        :param max_words: maximum number of words in a clip
        :param n_pair: number of clips to sample from each video
        :param bert_tokenizer: BERT tokenizer
        """
        self.csv = pd.read_csv(csv_path)
        self.caption = caption
        self.feature_path = features_path
        self.min_time = min_time
        self.max_time = max_time
        self.min_words = min_words
        self.max_words = max_words
        self.n_pair = n_pair
        self.bert_tokenizer = bert_tokenizer

    def __len__(self):
        return len(self.csv)

    def _get_text(self, caption, n_pair_max):
        n_caption = len(caption["start"])
        k = min(n_pair_max, n_caption)
        starts = np.zeros(k)
        ends = np.zeros(k)
        text = [""] * k
        r_ind = np.random.choice(range(n_caption), k, replace=False)

        for i in range(k):
            ind = r_ind[i]
            text[i], starts[i], ends[i] = self._get_single_text(caption, ind)

        text_embds = tokenize(
            text,
            self.bert_tokenizer,
            add_special_tokens=True,
            max_length=self.max_words,
            dynamic_padding=True,
            truncation=True,
        )

        return text_embds, text, starts, ends

    def _get_single_text(self, caption, ind):
        start, end = ind, ind
        words = str(caption["text"][ind])
        n_words = len(words.split(" "))
        diff = caption["end"][end] - caption["start"][start]
        while n_words < self.min_words or diff < self.min_time:
            if start > 0 and end < len(caption["end"]) - 1:
                next_words = str(caption["text"][end + 1])
                prev_words = str(caption["text"][start - 1])
                d1 = caption["end"][end + 1] - caption["start"][start]
                d2 = caption["end"][end] - caption["start"][start - 1]
                if (self.min_time > 0 and d2 <= d1) or (
                    self.min_time == 0 and len(next_words) <= len(prev_words)
                ):
                    start -= 1
                    words = prev_words + " " + words
                else:
                    end += 1
                    words = words + " " + next_words
            elif start > 0:
                prev_words = str(caption["text"][start - 1])
                words = prev_words + " " + words
                start -= 1
            elif end < len(caption["end"]) - 1:
                next_words = str(caption["text"][end + 1])
                words = words + " " + next_words
                end += 1
            else:
                break
            diff = caption["end"][end] - caption["start"][start]
            n_words = len(words.split(" "))
        return words, caption["start"][start], caption["end"][end]

    def _get_video(self, vid_path, s, e):
        feature_path = os.path.join(self.feature_path, vid_path)
        video = torch.from_numpy(np.load(feature_path)).float()
        video_len = torch.ones(len(s))
        output = torch.zeros(len(s), self.max_time, video.shape[-1])

        for i in range(len(s)):
            start = int(s[i])
            end = int(e[i]) + 1
            slice = video[start:end]
            assert len(slice) >= 1
            if len(slice) < self.max_time:
                video_len[i] = len(slice)
                output[i] = torch.cat(
                    [slice, torch.zeros(self.max_time - len(slice), slice.size(1))],
                    dim=0,
                )
            else:
                video_len[i] = self.max_time
                output[i] = slice[: self.max_time]

        return output, video_len

    def __getitem__(self, idx):
        video_id = self.csv["video_id"].values[idx]
        vid_path = self.csv["video_path"].values[idx]
        text, caption, starts, ends = self._get_text(
            self.caption[video_id], self.n_pair
        )
        video, video_len = self._get_video(vid_path, starts, ends)
        return {
            "video": video,
            "video_len": video_len,
            "text": text,
        }


def howto_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the text padded to the max length of the batch
    """
    bs = len(batch)
    video = torch.cat([batch[i]["video"] for i in range(bs)], 0)
    video_len = torch.cat([batch[i]["video_len"] for i in range(bs)], 0)
    text = [batch[i]["text"] for i in range(bs)]
    maxlen = max([x.shape[1] for x in text])
    text_padded = torch.zeros(sum(x.shape[0] for x in text), maxlen).long()
    idx = 0
    for i, tensor in enumerate(text):
        n, l = tensor.shape
        text_padded[idx : idx + n, :l] = tensor
        idx += n

    return {
        "video": video,
        "video_len": video_len,
        "text": text_padded,
    }
