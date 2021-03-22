import re
import torch
import torch.nn.functional as F
import json
import collections
import numpy as np

def tokenize(seq, tokenizer, add_special_tokens=True, max_length=10, dynamic_padding=True, truncation=True):
    """
    :param seq: sequence of sequences of text
    :param tokenizer: bert_tokenizer
    :return: torch tensor padded up to length max_length of bert tokens
    """
    tokens = tokenizer.batch_encode_plus(
            seq,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding="longest" if dynamic_padding else "max_length",
            truncation=truncation,
        )['input_ids']
    return torch.tensor(tokens, dtype=torch.long)

def compute_aggreeings(topk, answers, thresholds, names, metrics, ivqa=False):
    """ Updates metrics dictionary by computing aggreeings for different thresholds """
    if not ivqa:
        for i, x in enumerate(thresholds):
            agreeingsx = (topk[:, :x] == answers[:, :x]).sum().item()
            metrics[names[i]] += agreeingsx
    else:
        for i, x in enumerate(thresholds):
            predicted = F.one_hot(topk[:, :x], num_classes=answers.shape[-1]).sum(1)
            metrics[names[i]] += (predicted * answers).max(1)[0].sum().item()
    return metrics


class AverageMeter:
    """ Computes and stores the average and current value for training stats """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_mask(lengths, max_length):
    """ Computes a batch of padding masks given batched lengths """
    mask = 1 * (
        torch.arange(max_length).unsqueeze(1).to(lengths.device) < lengths
    ).transpose(0, 1)
    return mask


def compute_a2v(vocab_path, bert_tokenizer, amax_words):
    """ Precomputes GloVe answer embeddings for all answers in the vocabulary """
    a2id = json.load(open(vocab_path, "r"))
    id2a = {v: k for k, v in a2id.items()}
    a2v = tokenize(list(a2id.keys()), bert_tokenizer, add_special_tokens=True, max_length=amax_words,
                       dynamic_padding=True, truncation=True)
    if torch.cuda.is_available():
        a2v = a2v.cuda()  # (vocabulary_size, 1, we_dim)
    return a2id, id2a, a2v


def mask_tokens(inputs, tokenizer, mlm_probability):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
    )
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def get_types(dataset):
    """ Type2Id mapping for VideoQA datasets """
    if dataset == "tgif":
        return {"what": 0, "how": 1, "color": 2, "where": 3}
    elif dataset == "activitynet":
        return {
            "motion": 0,
            "spatial": 1,
            "temporal": 2,
            "yesno": 3,
            "color": 4,
            "object": 5,
            "location": 6,
            "number": 7,
            "other": 8,
        }
    elif dataset == "msvd" or dataset == "msrvtt":
        return {"what": 0, "how": 1, "color": 2, "where": 3, "who": 4, "when": 5}
    elif dataset == "ivqa":
        return {"scenes": 0}
    else:
        raise NotImplementedError


def get_most_common(loader, ivqa=False, n=4):
    """ Outputs most common answers and splits in n parts the answers depending on their frequency"""
    if ivqa:
        ans = []
        for a1, a2, a3, a4, a5 in zip(
            list(loader.dataset.data["answer1"]),
            list(loader.dataset.data["answer2"]),
            list(loader.dataset.data["answer3"]),
            list(loader.dataset.data["answer4"]),
            list(loader.dataset.data["answer5"]),
        ):
            counteri = collections.Counter([a1, a2, a3, a4, a5])
            for w in counteri:
                if (
                    counteri[w] >= 2
                ):  # an answer is considered as right if it has been annotated by two workers
                    ans.append(w)
    else:
        ans = list(loader.dataset.data["answer"])
    most_common = collections.Counter(ans).most_common()

    total = sum(x[1] for x in most_common)
    splits = [0] * (n + 1)
    j = 0
    for i in range(n):
        cur_total = 0
        while j < len(most_common) and cur_total < total / n:
            cur_total += most_common[j][1]
            j += 1
        splits[i + 1] = j
    return most_common, splits, total


def compute_word_stats(
    topk, answers, a2id, a2v, most_common, metrics, counts, ivqa, top10=False
):
    """ Similar as compute_agreeings, computes agreeings and counts for most common words """
    if not ivqa:
        for word, cword in most_common:
            if word not in a2id:
                counts[word] = cword
                continue
            predicted = topk[:, 0]
            metrics[f"acc_{word}"] += (
                (predicted[answers == a2id[word]] == a2id[word]).sum().item()
            )
            if top10:
                predicted10 = topk[:, :10]
                metrics[f"acc10_{word}"] += (
                    (predicted10[answers == a2id[word]] == a2id[word]).sum().item()
                )
            counts[word] += (answers == a2id[word]).sum().item()
    else:
        for word, cword in most_common:
            if word not in a2id:
                counts[word] = cword
                continue
            predicted = F.one_hot(topk[:, 0], num_classes=len(a2v))
            ans_word = answers[:, a2id[word]]
            metrics[f"acc_{word}"] += (
                (predicted[:, a2id[word]][ans_word == 1] * ans_word[ans_word == 1])
                .sum()
                .item()
            )
            if top10:
                predicted10 = F.one_hot(topk[:, :10], num_classes=len(a2v)).sum(1)
                metrics[f"acc10_{word}"] += (
                    (
                        predicted10[:, a2id[word]][ans_word == 1]
                        * ans_word[ans_word == 1]
                    )
                    .sum()
                    .item()
                )
            counts[word] += (ans_word == 1).sum().item()
    return metrics, counts

def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['R100'] = float(np.sum(ind < 100)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return metrics


def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r10 = metrics['R10']
    r100 = metrics['R100']
    mr = metrics['MR']
    return('R@1: {:.4f} - R@10: {:.4f} - R@100: {:.4f} - Median R: {}'.format(r1, r10, r100, mr))