import torch
import logging
import math
from tqdm import tqdm
from util import mask_tokens, get_mask, AverageMeter, compute_metrics, print_computed_metrics


def train_mlmcm(model, optimizer, dataloader, scheduler, epoch, args):
    model.train()
    running_mlm_loss, running_cm_loss = AverageMeter(), AverageMeter()
    for i, batch in enumerate(dataloader):
        tokens = batch["text"]
        text_mask = 1 * (tokens != 0).cuda()

        video = batch["video"].cuda()
        video_len = batch["video_len"].cuda()
        video_mask = get_mask(video_len, video.size(1)).cuda()

        inputs, labels = mask_tokens(
            tokens, model.module.bert.bert_tokenizer, mlm_probability=args.mlm_prob
        )
        inputs, labels, tokens = inputs.cuda(), labels.cuda(), tokens.cuda()

        mlm_loss = model(
            video,
            question=inputs,
            labels=labels,
            text_mask=text_mask,
            video_mask=video_mask,
            mode="mlm",
        )
        mlm_loss = mlm_loss.mean()
        cm_loss = model(
            video, question=tokens, text_mask=text_mask, video_mask=video_mask, mode="cm"
        )
        cm_loss = cm_loss.mean()
        loss = mlm_loss + cm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_mlm_loss.update(mlm_loss.detach().cpu().item())
        running_cm_loss.update(cm_loss.detach().cpu().item())
        if (i + 1) % (len(dataloader) // args.freq_display) == 0:
            logging.info(
                "Epoch %d, Epoch status: %.4f, Training MLM loss: %.4f, Training CM loss: %.4f"
                % (
                    epoch,
                    float(i + 1) / len(dataloader),
                    running_mlm_loss.avg,
                    running_cm_loss.avg,
                )
            )
            running_mlm_loss.reset()
            running_cm_loss.reset()

def eval_mlm(model, eval_dataloader, dataset_name, epoch):
    model.eval()
    loss = AverageMeter()
    with torch.no_grad():
        for i_batch, data in enumerate(eval_dataloader):
            tokens = data['text']
            inputs, labels = mask_tokens(tokens, model.module.bert.bert_tokenizer, mlm_probability=0.15)
            inputs, labels, tokens = inputs.cuda(), labels.cuda(), tokens.cuda()
            text_mask = 1 * (tokens != 0)
            video = data['video'].cuda()
            video_len = data['video_len'].cuda()
            video_mask = get_mask(video_len, video.size(1)).cuda()
            mlm_loss = model(video, inputs, labels=labels, text_mask=text_mask, video_mask=video_mask, mode='mlm')
            mlm_loss = mlm_loss.mean()
            loss.update(mlm_loss, len(tokens))
    logging.info(f"Epoch {epoch}, Val {dataset_name} MLM loss: {loss.avg:.4f}")

def eval_retrieval(model, eval_dataloader, dataset_name, epoch):
    model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(eval_dataloader):
            assert i_batch == 0 # evaluation done in one batch
            tokens = data['text'].cuda()
            text_mask = 1 * (tokens != 0)
            video = data['video'].cuda()
            video_len = data['video_len'].cuda()
            video_mask = get_mask(video_len, video.size(1)).cuda()
            m = torch.zeros(len(tokens), len(video)).cuda()
            n_gpus = torch.cuda.device_count()
            video_rep = video.repeat(n_gpus, 1, 1) # repeat so that on each gpu there are all videos
            video_mask_rep = video_mask.repeat(n_gpus, 1)
            for j in tqdm(range(math.ceil(len(tokens) // n_gpus))): # one text passed with all videos on each gpu
                tokens_one = tokens[j * n_gpus: (j + 1) * n_gpus]
                text_mask_one = text_mask[j * n_gpus: (j + 1) * n_gpus]
                output = model(video_rep, tokens_one, text_mask = text_mask_one, video_mask = video_mask_rep, mode = 'retrieval')
                m[j * n_gpus: (j + 1) * n_gpus] = output.view(n_gpus, -1)
            m  = m.detach().cpu().numpy()
            metrics = compute_metrics(m)
    logging.info(f"Epoch {epoch}, Val {dataset_name} Text-Video Retrieval: " + print_computed_metrics(metrics))