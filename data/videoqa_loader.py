import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import collections
from util import tokenize


class VideoQADataset(Dataset):
    def __init__(
        self,
        csv_path,
        features,
        qmax_words=20,
        amax_words=5,
        bert_tokenizer=None,
        a2id=None,
        ivqa=False,
        max_feats=20,
        mc=0
    ):
        """
        :param csv_path: path to a csv containing columns video_id, question, answer
        :param features: dictionary mapping video_id to torch tensor of features
        :param qmax_words: maximum number of words for a question
        :param amax_words: maximum number of words for an answer
        :param bert_tokenizer: BERT tokenizer
        :param a2id: answer to index mapping
        :param ivqa: whether to use iVQA or not
        :param max_feats: maximum frames to sample from a video
        """
        self.data = pd.read_csv(csv_path)
        self.features = features
        self.qmax_words = qmax_words
        self.amax_words = amax_words
        self.a2id = a2id
        self.bert_tokenizer = bert_tokenizer
        self.ivqa = ivqa
        self.max_feats = max_feats
        self.mc = mc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid_id = self.data["video_id"].values[index]
        video = self.features[vid_id]
        if len(video) < self.max_feats:
            video = video[: self.max_feats]
            vid_duration = len(video)
            if len(video) < self.max_feats:
                video = torch.cat(
                    [video, torch.zeros(self.max_feats - len(video), video.shape[1])]
                )
        else:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            vid_duration = len(video)

        type, answer, answer_len = 0, 0, 0
        if self.ivqa:
            answer_txt = collections.Counter(
                [
                    self.data["answer1"].values[index],
                    self.data["answer2"].values[index],
                    self.data["answer3"].values[index],
                    self.data["answer4"].values[index],
                    self.data["answer5"].values[index],
                ]
            )
            answer_id = torch.zeros(len(self.a2id))
            for x in answer_txt:
                if x in self.a2id:
                    answer_id[self.a2id[x]] = answer_txt[x]
            answer_txt = ", ".join([str(x) + "(" + str(answer_txt[x]) + ")" for x in answer_txt])
        elif self.mc:
            answer_id = int(self.data["answer"][index])
            answer_txt = [self.data["a" + str(i + 1)][index] for i in range(self.mc)]
            answer = tokenize(answer_txt, self.bert_tokenizer, add_special_tokens=True, max_length=self.amax_words,
                               dynamic_padding=True, truncation=True)
        else:
            answer_txt = self.data["answer"].values[index]
            answer_id = self.a2id.get(
                answer_txt, -1
            )  # put an answer_id -1 if not in top answers, that will be considered wrong during evaluation
        if not self.mc:
            type = self.data["type"].values[index]

        question_txt = self.data["question"][index]
        question_embd = torch.tensor(
            self.bert_tokenizer.encode(
                question_txt,
                add_special_tokens=True,
                padding="longest",
                max_length=self.qmax_words,
                truncation=True,
            ),
            dtype=torch.long,
        )

        return {
            "video_id": vid_id,
            "video": video,
            "video_len": vid_duration,
            "question": question_embd,
            "question_txt": question_txt,
            "type": type,
            "answer_id": answer_id,
            "answer_txt": answer_txt,
            "answer": answer,
        }

def videoqa_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the question and the ans candidates padded to the max length of the batch
    """
    qmax_len = max(len(batch[i]["question"]) for i in range(len(batch)))
    for i in range(len(batch)):
        if len(batch[i]["question"]) < qmax_len:
            batch[i]["question"] = torch.cat(
                [
                    batch[i]["question"],
                    torch.zeros(qmax_len - len(batch[i]["question"]), dtype=torch.long),
                ],
                0,
            )

    if not isinstance(batch[0]["answer"], int):
        amax_len = max(x["answer"].size(1) for x in batch)
        for i in range(len(batch)):
            if batch[i]["answer"].size(1) < amax_len:
                batch[i]["answer"] = torch.cat(
                    [
                        batch[i]["answer"],
                        torch.zeros(
                            (
                                batch[i]["answer"].size(0),
                                amax_len - batch[i]["answer"].size(1),
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    1,
                )

    return default_collate(batch)


def get_videoqa_loaders(args, features, a2id, bert_tokenizer):
    train_dataset = VideoQADataset(
        csv_path=args.train_csv_path,
        features=features,
        qmax_words=args.qmax_words,
        amax_words=args.amax_words,
        bert_tokenizer=bert_tokenizer,
        a2id=a2id,
        ivqa=(args.dataset == "ivqa"),
        max_feats=args.max_feats,
        mc=args.mc
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        shuffle=True,
        drop_last=True,
        collate_fn=videoqa_collate_fn
    )

    test_dataset = VideoQADataset(
        csv_path=args.test_csv_path,
        features=features,
        qmax_words=args.qmax_words,
        amax_words=args.amax_words,
        bert_tokenizer=bert_tokenizer,
        a2id=a2id,
        ivqa=(args.dataset == "ivqa"),
        max_feats=args.max_feats,
        mc=args.mc
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
        collate_fn=videoqa_collate_fn
    )

    val_dataset = VideoQADataset(
            csv_path=args.val_csv_path,
            features=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            bert_tokenizer=bert_tokenizer,
            a2id=a2id,
            ivqa=(args.dataset == "ivqa"),
            max_feats=args.max_feats,
            mc=args.mc
    )

    val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            collate_fn=videoqa_collate_fn
        )
    return (
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
        test_dataset,
        test_loader,
    )
