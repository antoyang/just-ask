import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import random
import os
import pickle
import logging
from args import get_args
from model.multimodal_transformer import MMT_VideoQA
from loss import Contrastive_Loss
from data.howtovqa_loader import HowToVQA_Dataset, howtovqa_collate_fn
from data.webvidvqa_loader import WebVidVQA_Dataset
from train.train_howtovqa import train_howtovqa, eval_howtovqa
from transformers import DistilBertTokenizer

# args, logging
args = get_args()
assert args.checkpoint_dir
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
model = nn.DataParallel(model)
if args.pretrain_path != "":
    model.load_state_dict(torch.load(args.pretrain_path))
    logging.info(f"Loaded checkpoint {args.pretrain_path}")
model.cuda()
logging.info("Using {} GPUs".format(torch.cuda.device_count()))
logging.info(
    f"Nb of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

# Load captions, dataloaders
bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

if args.dataset == "howtovqa":
    with open(args.caption_path, "rb") as caption_file:
        caption = pickle.load(caption_file)
    logging.info("Pickle loaded")
    trainset = HowToVQA_Dataset(
        csv_path=args.train_csv_path,
        caption=caption,
        features_path=args.features_path,
        qmax_words=args.qmax_words,
        amax_words=args.amax_words,
        train=True,
        n_pair=args.n_pair,
        bert_tokenizer=bert_tokenizer,
        max_feats=args.max_feats,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        shuffle=True,
        drop_last=True,
        collate_fn=howtovqa_collate_fn,
    )

    valset = HowToVQA_Dataset(
        csv_path=args.val_csv_path,
        caption=caption,
        features_path=args.features_path,
        qmax_words=args.qmax_words,
        amax_words=args.amax_words,
        train=False,
        n_pair=args.n_pair,
        bert_tokenizer=bert_tokenizer,
        max_feats=args.max_feats,
    )

    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
        collate_fn=howtovqa_collate_fn,
    )
elif args.dataset == "webvidvqa":
    with open(args.webvidvqa_caption_path, "rb") as caption_file:
        caption = pickle.load(caption_file)
    logging.info("Pickle loaded")

    trainset = WebVidVQA_Dataset(
        csv_path=args.train_csv_path,
        caption=caption,
        features_path=args.features_path,
        qmax_words=args.qmax_words,
        amax_words=args.amax_words,
        train=True,
        n_pair=1,
        bert_tokenizer=bert_tokenizer,
        max_feats=args.max_feats,
        feature_dim=args.feature_dim,
    )
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        shuffle=True,
        drop_last=True,
        collate_fn=howtovqa_collate_fn,
    )

    valset = WebVidVQA_Dataset(
        csv_path=args.val_csv_path,
        caption=caption,
        features_path=args.features_path,
        qmax_words=args.qmax_words,
        amax_words=args.amax_words,
        train=False,
        n_pair=1,
        bert_tokenizer=bert_tokenizer,
        max_feats=args.max_feats,
        feature_dim=args.feature_dim,
    )

    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size_val * args.n_pair,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
        collate_fn=howtovqa_collate_fn,
    )

logging.info("number of train videos: {}".format(len(train_loader.dataset)))
logging.info("number of val videos: {}".format(len(val_loader.dataset)))

# Loss, Optimizer, Scheduler
criterion = Contrastive_Loss()
criterion.cuda()
params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
optimizer = optim.Adam(
    params_for_optimization,
    lr=args.lr,
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 0, len(train_loader) * args.epochs
)

# Train
for epoch in range(args.epochs):
    eval_howtovqa(model, val_loader, args)
    train_howtovqa(model, train_loader, optimizer, criterion, scheduler, epoch, args)
    torch.save(model.state_dict(), os.path.join(args.save_dir, f"e{epoch}.pth"))
eval_howtovqa(model, val_loader, args)
