import pickle
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import logging
from train.train_nce import train_ptva, eval_retrieval
from args import get_args
from data.howto_loader import HowTo_Dataset, howto_collate_fn
from data.videotext_loader import Youcook_Dataset, videotext_collate_fn, VideoText_Dataset
from transformers import DistilBertTokenizer
from model.multimodal_transformer import MMT_VideoQA
from loss import NCELoss

# args, logging
args = get_args()
assert args.checkpoint_dir
if not (os.path.isdir(args.save_dir)):
    os.mkdir(args.save_dir)
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
    baseline="video",
)  # pretraining without question input

model = nn.DataParallel(model)
model.cuda()

# Load captions, dataloaders
caption = pickle.load(open(args.caption_path, "rb"))
logging.info("Pickle loaded")
bert_tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

dataset = HowTo_Dataset(
    csv_path=args.train_csv_path,
    caption=caption,
    features_path=args.features_path,
    min_time=args.min_time,
    max_time=args.max_feats,
    max_words=args.qmax_words,
    min_words=args.min_words,
    n_pair=args.n_pair,
    bert_tokenizer=bert_tokenizer
)

dataset_size = len(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_thread_reader,
    shuffle=True,
    batch_sampler=None,
    drop_last=True,
    collate_fn=howto_collate_fn
)

youcook_dataset = Youcook_Dataset(
        data=args.youcook_val_path,
        max_words=args.qmax_words,
        bert_tokenizer = bert_tokenizer,
        max_feats=args.max_feats,
    )
youcook_loader = DataLoader(
        youcook_dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        collate_fn=videotext_collate_fn
    )

msrvtt_dataset = VideoText_Dataset(
        csv_path=args.msrvtt_test_csv_path,
        features_path=args.msrvtt_test_features_path,
        max_words=args.qmax_words,
        bert_tokenizer = bert_tokenizer,
        max_feats = args.max_feats,
    )
msrvtt_loader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
        collate_fn=videotext_collate_fn
    )

# Optimizer, Scheduler
criterion = NCELoss(batch_size=args.n_pair * args.batch_size)
criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=len(dataloader), gamma=args.lr_decay)

for epoch in range(args.epochs):
    eval_retrieval(model, youcook_loader, 'YouCook2', epoch)
    eval_retrieval(model, msrvtt_loader, 'MSR-VTT', epoch)
    train_ptva(model, optimizer, dataloader, criterion, scheduler, epoch, args)
    torch.save(model.state_dict(), os.path.join(args.save_dir, f"e{epoch}.pth"))
eval_retrieval(model, youcook_loader, 'YouCook2', args.epochs)
eval_retrieval(model, msrvtt_loader, 'MSR-VTT', args.epochs)
