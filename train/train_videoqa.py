import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import collections
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens


def eval(model, val_loader, a2v, args, test=False):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)

    with torch.no_grad():
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
        for i, batch in enumerate(val_loader):
            answer_id, answer, video, question = (
                batch["answer_id"],
                batch["answer"],
                batch["video"].cuda(),
                batch["question"].cuda(),
            )
            video_len = batch["video_len"]
            question_mask = (question > 0).float()
            video_mask = get_mask(video_len, video.size(1)).cuda()
            count += answer_id.size(0)

            if not args.mc:
                predicts = model(
                    video, question, text_mask=question_mask, video_mask=video_mask,
                )
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
                fusion_proj, answer_proj = model(
                    video, question, text_mask=question_mask, video_mask=video_mask,
                    answer=answer.cuda()
                )
                fusion_proj = fusion_proj.unsqueeze(2)
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze()
                predicted = torch.max(predicts, dim=1).indices.cpu()
                metrics["acc"] += (predicted == answer_id).sum().item()

    step = "val" if not test else "test"
    for k in metrics:
        v = metrics[k] / count
        logging.info(f"{step} {k}: {v:.2%}")

    return metrics["acc"] / count


def train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args):
    model.train()
    running_vqa_loss, running_acc, running_mlm_loss = AverageMeter(), AverageMeter(), AverageMeter()
    for i, batch in enumerate(train_loader):
        answer_id, answer, video, question = (
            batch["answer_id"],
            batch["answer"],
            batch["video"].cuda(),
            batch["question"].cuda(),
        )
        video_len = batch["video_len"]
        question_mask = (question > 0).float()
        video_mask = (
            get_mask(video_len, video.size(1)).cuda() if args.max_feats > 0 else None
        )
        N = answer_id.size(0)
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
            predicts = model(
                video, question, text_mask=question_mask, video_mask=video_mask,
            )
        else:
            fusion_proj, answer_proj = model(
                video, question, text_mask=question_mask, video_mask=video_mask,
                answer=answer.cuda()
            )
            fusion_proj = fusion_proj.unsqueeze(2)
            predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

        if args.dataset == "ivqa":
            a = (answer_id / 2).clamp(max=1).cuda()
            vqa_loss = criterion(predicts, a)
            predicted = torch.max(predicts, dim=1).indices.cpu()
            predicted = F.one_hot(predicted, num_classes=len(a2v))
            running_acc.update((predicted * a.cpu()).sum().item() / N, N)
        else:
            vqa_loss = criterion(predicts, answer_id.cuda())
            predicted = torch.max(predicts, dim=1).indices.cpu()
            running_acc.update((predicted == answer_id).sum().item() / N, N)

        if args.mlm_prob:
            inputs = batch["question"]
            inputs, labels = mask_tokens(
                inputs, model.module.bert.bert_tokenizer, mlm_probability=0.15
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
                    f"Epoch {epoch + 1}, Epoch status: {float(i + 1) / len(train_loader):.4f}, Training VideoQA loss: "
                    f"{running_vqa_loss.avg:.4f}, Training acc: {running_acc.avg:.2%}, Training MLM loss: {running_mlm_loss.avg:.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch + 1}, Epoch status: {float(i + 1) / len(train_loader):.4f}, Training VideoQA loss: "
                    f"{running_vqa_loss.avg:.4f}, Training acc: {running_acc.avg:.2%}"
                )
            running_acc.reset()
            running_vqa_loss.reset()
            running_mlm_loss.reset()
