import logging
import torch
from util import get_mask, AverageMeter, print_computed_metrics, compute_metrics


def train_ptva(model, optimizer, dataloader, criterion, scheduler, epoch, args):
    model.train()
    running_loss = AverageMeter()
    for i, batch in enumerate(dataloader):
        text = batch["text"].cuda()
        video = batch["video"].cuda()
        video_len = batch["video_len"].cuda()
        video_mask = get_mask(video_len, video.size(1)).cuda()

        text, video = model(video, answer=text, video_mask=video_mask)
        sim_matrix = text @ (video.t())
        loss = criterion(sim_matrix)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss.update(loss)
        if (i + 1) % (len(dataloader) // args.freq_display) == 0:
            logging.info(
                "Epoch %d, Epoch status: %.4f, Training loss: %.4f"
                % (epoch, float(i + 1) / len(dataloader), running_loss.avg)
            )
            running_loss.reset()

def eval_retrieval(model, eval_dataloader, dataset_name, epoch):
    model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(eval_dataloader):
            assert i_batch == 0 # evaluation done in one batch
            text = data['text'].cuda()
            video = data['video'].cuda()
            video_len = data['video_len'].cuda()
            video_mask = get_mask(video_len, video.size(1)).cuda()
            text, video = model(video, answer=text, video_mask=video_mask)
            sim_matrix = text @ (video.t())
            sim_matrix = sim_matrix.detach().cpu().numpy()
            metrics = compute_metrics(sim_matrix)
    logging.info(f"Epoch {epoch}, Val {dataset_name} Text-Video Retrieval: " + print_computed_metrics(metrics))