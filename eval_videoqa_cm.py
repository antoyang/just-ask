import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import random
import os
import logging
import collections
import pandas as pd
from transformers import DistilBertTokenizer
from args import get_args
from model.multimodal_transformer import MMT_VideoQA
from util import compute_a2v, tokenize, get_mask, compute_aggreeings
from tqdm import tqdm


class VideoQADataset(Dataset):
    def __init__(
        self,
        csv_path,
        features,
        qmax_words=20,
        bert_tokenizer=None,
        a2id=None,
        ivqa=False,
        max_feats=20,
        tmp_sample=0,
        id2a=None,
        mc=0,
    ):
        self.data = pd.read_csv(csv_path)
        self.features = features
        self.qmax_words = qmax_words
        self.a2id = a2id
        self.bert_tokenizer = bert_tokenizer
        self.ivqa = ivqa
        self.max_feats = max_feats
        self.tmp_sample = tmp_sample
        self.id2a = id2a
        self.mc = mc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid_id = self.data["video_id"].values[index]
        video = self.features[vid_id]
        if len(video) < self.max_feats or not self.tmp_sample:
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
            answer_txt = ", ".join(
                [str(x) + "(" + str(answer_txt[x]) + ")" for x in answer_txt]
            )
        elif self.mc:
            answer_id = int(self.data["answer"][index])
            answer_txt = [self.data["a" + str(i + 1)][index] for i in range(self.mc)]
            question_txt = self.data["question"][index]
            qa_txt = [
                question_txt + " " + x for x in answer_txt
            ]  # concatenate question with each possible answer
            question_embd = tokenize(
                qa_txt,
                self.bert_tokenizer,
                add_special_tokens=True,
                max_length=self.qmax_words,
                dynamic_padding=True,
                truncation=True,
            )
        else:
            answer_txt = self.data["answer"].values[index]
            answer_id = self.a2id.get(
                answer_txt, -1
            )  # put an answer_id -1 if not in top answers, that will be considered wrong during evaluation

        if not self.mc:
            question_txt = self.data["question"][index]
            qa_txt = [
                question_txt + " " + self.id2a[i] for i in range(len(self.id2a))
            ]  # concatenate question with each possible answer
            question_embd = tokenize(
                qa_txt,
                self.bert_tokenizer,
                add_special_tokens=True,
                max_length=self.qmax_words,
                dynamic_padding=True,
                truncation=True,
            )

        return {
            "video_id": vid_id,
            "video": video,
            "video_len": vid_duration,
            "question": question_embd,
            "answer_id": answer_id,
        }


def videoqa_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the question and the ans candidates padded to the max length of the batch
    """
    bs = len(batch)
    que = [batch[i]["question"] for i in range(bs)]
    maxquelen = max([x.shape[-1] for x in que])
    nans = que[0].shape[0]
    question = torch.zeros(bs, nans, maxquelen).long()
    for i, tensor in enumerate(que):
        n, l = tensor.shape
        question[i, :, :l] = tensor

    return {
        "video_id": default_collate([batch[i]["video_id"] for i in range(bs)]),
        "video": default_collate([batch[i]["video"] for i in range(bs)]),
        "video_len": default_collate([batch[i]["video_len"] for i in range(bs)]),
        "question": question,
        "answer_id": default_collate([batch[i]["answer_id"] for i in range(bs)]),
    }


def eval(model, val_loader, args, test=False):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)
    vid2ans = {}
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            answer_id, video, question = (
                batch["answer_id"].squeeze(),
                batch["video"].cuda(),
                batch["question"].cuda(),
            )
            video_len = batch["video_len"]
            question_mask = (question > 0).float()
            video_mask = get_mask(video_len, video.size(1)).cuda()
            count += answer_id.size(0)

            predicts = model(
                video,
                question,
                text_mask=question_mask,
                video_mask=video_mask,
                mode="vqacm",
            )
            predicts = predicts.view(answer_id.size(0), -1)
            if not args.mc:
                topk = torch.topk(predicts, dim=1, k=10).indices.cpu()
                if args.dataset != "ivqa":
                    answer_id_expanded = answer_id.view(-1, 1).expand_as(topk)
                else:
                    answer_id = (answer_id / 2).clamp(max=1)
                    answer_id_expanded = answer_id

                metrics = compute_aggreeings(
                    topk,
                    answer_id_expanded,
                    [1, 10],
                    ["acc", "acc10"],
                    metrics,
                    ivqa=(args.dataset == "ivqa"),
                )
            else:
                predicted = torch.max(predicts, dim=1).indices.cpu()
                metrics["acc"] += (predicted == answer_id).sum().item()

            """video_id = batch["video_id"]
            top1 = topk[:, 0]
            for k in range(len(video_id)):
                vid2ans[video_id[k]] = id2a[top1[k].item()]"""

    step = "val" if not test else "test"
    for k in metrics:
        v = metrics[k] / count
        logging.info(f"{step} {k}: {v:.2%}")
    # pickle.dump(vid2ans, open(os.path.join(args.save_dir, "preds.pkl"), 'wb'))

    return metrics["acc"] / count


# args, logging
args = get_args()
if not (os.path.isdir(args.save_dir)):
    os.mkdir(os.path.join(args.save_dir))
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logFormatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
rootLogger = logging.getLogger()
fileHandler = logging.FileHandler(os.path.join(args.save_dir, "stdout.log"), "w+")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
logging.info(args)

# set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# get answer embeddings
bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
a2id, id2a, a2v = None, None, None
if not args.mc:
    a2id, id2a, a2v = compute_a2v(
        vocab_path=args.vocab_path,
        bert_tokenizer=bert_tokenizer,
        amax_words=args.amax_words,
    )
    logging.info(f"Length of Answer Vocabulary: {len(a2id)}")

# Model
model = MMT_VideoQA(
    feature_dim=args.feature_dim,
    word_dim=args.word_dim,
    N=args.n_layers,
    d_model=args.embd_dim,
    d_ff=args.ff_dim,
    h=args.n_heads,
    dropout=args.dropout,
    T=args.max_feats,
    Q=args.qmax_words,
    baseline=args.baseline,
)
model.cuda()
logging.info("Using {} GPUs".format(torch.cuda.device_count()))

# Load pretrain path
model = nn.DataParallel(model)
if args.pretrain_path != "":
    model.load_state_dict(torch.load(args.pretrain_path))
logging.info(
    f"Nb of trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

# Dataloaders
features = torch.load(args.features_path)
test_dataset = VideoQADataset(
    csv_path=args.test_csv_path,
    features=features,
    qmax_words=args.qmax_words,
    bert_tokenizer=bert_tokenizer,
    a2id=a2id,
    ivqa=(args.dataset == "ivqa"),
    max_feats=args.max_feats,
    id2a=id2a,
    mc=args.mc,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=torch.cuda.device_count(),
    num_workers=args.num_thread_reader,
    shuffle=False,
    drop_last=False,
    collate_fn=videoqa_collate_fn,
)

logging.info("number of test instances: {}".format(len(test_loader.dataset)))

# Zero-shot VideoQA with cross-modal matching module
eval(model, test_loader, args, test=True)
