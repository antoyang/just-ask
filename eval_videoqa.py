import torch
import torch.nn as nn
import numpy as np
import random
import collections
from args import get_args
from model.multimodal_transformer import MMT_VideoQA
from util import (
    compute_a2v,
    get_mask,
    compute_aggreeings,
    get_types,
    get_most_common,
    compute_word_stats,
)
from data.videoqa_loader import get_videoqa_loaders
from transformers import DistilBertTokenizer


def eval(model, val_loader, a2v, args, types, most_common, splits, total):
    count = 0
    metrics, counts, metrics_word, counts_word = (
        collections.defaultdict(int),
        collections.defaultdict(int),
        collections.defaultdict(int),
        collections.defaultdict(int),
    )

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            answer_id, video, question = (
                batch["answer_id"],
                batch["video"].cuda(),
                batch["question"].cuda(),
            )
            video_len = batch["video_len"]
            type = batch["type"]
            question_mask = (question > 0).float()
            video_mask = get_mask(video_len, video.size(1)).cuda()
            count += answer_id.size(0)

            predicts = model(
                video, question, text_mask=question_mask, video_mask=video_mask
            )

            topk = torch.topk(predicts, dim=1, k=10).indices.cpu()
            if args.dataset != "ivqa":
                answer_id_expanded = answer_id.view(-1, 1).expand_as(topk)
            else:
                answer_id = (answer_id / 2).clamp(max=1)
                answer_id_expanded = answer_id
            for x, y in types.items():  # compute per type VideoQA stats
                counts[x] += sum(type == y).item()
                metrics = compute_aggreeings(
                    topk[type == y],
                    answer_id_expanded[type == y],
                    [1, 10],
                    [x + "/acc", x + "/acc10"],
                    metrics,
                    ivqa=(args.dataset == "ivqa"),
                )

            # compute per word VideoQA stats
            metrics_word, counts_word = compute_word_stats(
                topk,
                answer_id.cpu(),
                a2id,
                a2v,
                most_common,
                metrics_word,
                counts_word,
                ivqa=(args.dataset == "ivqa"),
                top10=True,
            )

    for k in range(1, len(splits)):  # compute per splits VideoQA stats
        agreeings_splitk = sum(
            metrics_word["acc_" + w[0]]
            for it, w in enumerate(most_common)
            if it >= splits[k - 1] and it < splits[k]
        )
        agreeings10_splitk = sum(
            metrics_word["acc10_" + w[0]]
            for it, w in enumerate(most_common)
            if it >= splits[k - 1] and it < splits[k]
        )
        counts_splitk = sum(
            counts_word[w[0]]
            for it, w in enumerate(most_common)
            if it >= splits[k - 1] and it < splits[k]
        )
        print(
            f"split {k}: {counts_splitk / total: .4f}, {agreeings_splitk / counts_splitk:.2%}, {agreeings10_splitk / counts_splitk:.2%}"
        )

    for x in types:  # deduce from types stats the global stats
        metrics["acc"] += metrics[x + "/acc"]
        metrics["acc10"] += metrics[x + "/acc10"]

    for k in metrics:
        if "/" in k:
            v = metrics[k] / counts[k.split("/")[0]]
            print(f"test {k}: {v:.2%}")
        else:
            v = metrics[k] / count
            print(f"test {k}: {v:.2%}")

    return metrics["acc"] / count


# args
args = get_args()
assert args.pretrain_path

# set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# get answer embeddings
bert_tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
a2id, id2a, a2v = None, None, None
if not args.mc:
    a2id, id2a, a2v = compute_a2v(
        vocab_path=args.vocab_path, bert_tokenizer=bert_tokenizer, amax_words=args.amax_words
    )

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

# Load pretrain path
model = nn.DataParallel(model)
model.load_state_dict(torch.load(args.pretrain_path))
model.eval()
with torch.no_grad():
    model.module._compute_answer_embedding(a2v)

# Dataloaders
features = torch.load(args.features_path)
_, _, _, _, test_dataset, test_loader = get_videoqa_loaders(
    args, features, a2id, bert_tokenizer
)

types = get_types(args.dataset)
most_common, splits, total = get_most_common(test_loader, ivqa=(args.dataset == "ivqa"))

eval(model, test_loader, a2v, args, types, most_common, splits, total)
