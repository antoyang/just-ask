import torch
import torch.nn as nn
import logging
import collections
import numpy as np
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens


def eval_sqa(model, val_loader, args):
    model.eval()
    metrics = collections.defaultdict(int)
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            answer, video, question = (
                batch["answer"].cuda(),
                batch["video"].cuda(),
                batch["question"].cuda(),
            )
            video_len = batch["video_len"].squeeze()
            video = video.squeeze()
            video_mask = get_mask(video_len, video.size(1)).cuda()
            question_mask = (question > 0).float()
            count += answer.size(0)
            atxt_unique, ans_idx, ans_inv = np.unique(batch['atxt'], return_index=True,
                                                          return_inverse=True)  # only keep unique answers
            answer = answer[ans_idx]

            fusion_proj, answer_proj = model(
                    video,
                    question=question,
                    answer=answer,
                    text_mask=question_mask,
                    video_mask=video_mask,
            )
            predicts = fusion_proj @ (answer_proj.t())

            topk = torch.topk(predicts, dim=1, k=10).indices.cpu()
            answer_id_expanded = (
                torch.from_numpy(ans_inv).view(-1, 1).expand_as(topk)
            )
            metrics = compute_aggreeings(
                topk,
                answer_id_expanded,
                [1, 10],
                ["rec", "rec10"],
                metrics,
            )

            if args.mlm_prob:
                inputs, labels = mask_tokens(
                    question.cpu(), model.module.bert.bert_tokenizer, mlm_probability=0.15
                )
                mlm_loss = model(
                    video,
                    question=inputs.cuda(),
                    labels=labels.cuda(),
                    text_mask=question_mask,
                    video_mask=video_mask,
                    mode="mlm",
                )
                mlm_loss = mlm_loss.mean()
                metrics["mlm_loss"] += mlm_loss * answer.size(0)

    for k in metrics:
        v = metrics[k] / count
        if "mlm" in k:
            logging.info(f"val {k}: {v:.4f}")
        else:
            logging.info(f"val {k}: {v:.2%}")

    return metrics["rec"] / count


def train_sqa(model, train_loader, optimizer, criterion, scheduler, epoch, args):
    model.train()
    running_vqa_loss, running_mlm_loss = AverageMeter(), AverageMeter()
    for i, batch in enumerate(train_loader):
        answer, video, question = (
            batch["answer"].cuda(),
            batch["video"].cuda(),
            batch["question"].cuda(),
        )
        video_len = batch["video_len"]
        video_mask = get_mask(video_len, video.size(1)).cuda()
        question_mask = (question > 0).float()
        N = answer.size(0)
        atxt_unique, ans_idx, ans_inv = np.unique(batch['atxt'], return_index=True, return_inverse=True) # only keep unique answers
        answer = answer[ans_idx]

        fusion_proj, answer_proj = model(
                video,
                question=question,
                answer=answer,
                text_mask=question_mask,
                video_mask=video_mask,
        )
        predicts = fusion_proj @ (answer_proj.t())
        target = torch.from_numpy(ans_inv).cuda()
        vqa_loss = criterion(predicts, target)

        if args.mlm_prob:
            inputs, labels = mask_tokens(
                question.cpu(), model.module.bert.bert_tokenizer, mlm_probability=0.15
            )
            mlm_loss = model(
                video,
                question=inputs.cuda(),
                labels=labels.cuda(),
                text_mask=question_mask,
                video_mask=video_mask,
                mode="mlm",
            )
            mlm_loss = mlm_loss.mean()
            loss = mlm_loss + vqa_loss
        else:
            loss = vqa_loss

        optimizer.zero_grad()
        loss.backward()
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()
        scheduler.step()

        running_vqa_loss.update(vqa_loss.detach().cpu().item(), N)
        if args.mlm_prob:
            running_mlm_loss.update(mlm_loss.detach().cpu().item(), N)
        if (i + 1) % (len(train_loader) // args.freq_display) == 0:
            if args.mlm_prob:
                logging.info(
                    f"Epoch {epoch + 1}, Epoch status: {float(i + 1) / len(train_loader):.4f}, Training VQA loss: {running_vqa_loss.avg:.4f}, "
                    f"Training MLM loss: {running_mlm_loss.avg:.4f}"
                )
            else:
                logging.info(
                        f"Epoch {epoch + 1}, Epoch status: {float(i + 1) / len(train_loader):.4f}, Training loss: {running_vqa_loss.avg:.4f}")
            running_vqa_loss.reset()
            running_mlm_loss.reset()
