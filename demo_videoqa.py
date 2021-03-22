import torch
import torch.nn as nn
import numpy as np
import random
from transformers import DistilBertTokenizer
from args import get_args
from model.multimodal_transformer import MMT_VideoQA
from util import compute_a2v, get_mask
import ffmpeg
from extract.s3dg import S3D
from extract.preprocessing import Preprocessing
from global_parameters import S3D_PATH

# args
args = get_args()
assert args.pretrain_path
assert args.question_example
assert args.video_example

# set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# get answer embeddings
bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
a2id, id2a, a2v = compute_a2v(
    vocab_path=args.vocab_path,
    bert_tokenizer=bert_tokenizer,
    amax_words=args.amax_words,
)
print(f"Length of Answer Vocabulary: {len(a2id)}")

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
model.module._compute_answer_embedding(a2v)

question_txt = args.question_example
video_path = args.video_example

# Tokenize Question
question = torch.tensor(
    model.module.bert.bert_tokenizer.encode(
        question_txt,
        add_special_tokens=True,
        padding="max_length",
        max_length=args.qmax_words,
        truncation=True,
    ),
    dtype=torch.long,
)
question = question.cuda().unsqueeze(0)
question_mask = question > 0

# Video Extractor
video_extractor = S3D(512, space_to_depth=True, embd=1, feature_map=0)
video_extractor.load_state_dict(torch.load(S3D_PATH))
video_extractor.eval()
video_extractor = torch.nn.DataParallel(video_extractor)
video_extractor = video_extractor.cuda()
preprocess = Preprocessing(num_frames=16)


with torch.no_grad():
    # Extract Video Feature
    probe = ffmpeg.probe(video_path)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    num, denum = video_stream["avg_frame_rate"].split("/")
    frame_rate = int(num) / int(denum)
    if height >= width:
        h, w = int(height * 224 / width), 224
    else:
        h, w = 224, int(width * 224 / height)
    assert frame_rate >= 1

    cmd = ffmpeg.input(video_path).filter("fps", fps=16).filter("scale", w, h)
    x = int((w - 224) / 2.0)
    y = int((h - 224) / 2.0)
    cmd = cmd.crop(x, y, 224, 224)
    out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
        capture_stdout=True, quiet=True
    )

    h, w = 224, 224
    video = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
    video = torch.from_numpy(video.astype("float32"))
    video = video.permute(0, 3, 1, 2)
    video = video.squeeze().cuda()
    video = preprocess(video)
    video = video_extractor(video)

    # Pad Video
    if len(video) < args.max_feats:
        video = video[: args.max_feats]
        video_len = len(video)
        if len(video) < args.max_feats:
            video = torch.cat(
                [video, torch.zeros(args.max_feats - len(video), video.shape[1]).cuda()]
            )
    else:
        sampled = []
        for j in range(args.max_feats):
            sampled.append(video[(j * len(video)) // args.max_feats])
        video = torch.stack(sampled)
        video_len = len(video)

    video_len = torch.Tensor([video_len])
    video = video.unsqueeze(0)
    video_mask = get_mask(video_len, video.size(1)).cuda()

    # Get Predictions
    predicts = model(
        video, question=question, text_mask=question_mask, video_mask=video_mask
    )
    topk = torch.topk(predicts, dim=1, k=5)
    topk_txt = [[id2a[x.item()] for x in y] for y in topk.indices.cpu()]
    topk_scores = [[f"{x:.2f}".format() for x in y] for y in topk.values.cpu()]
    topk_all = [
        [x + "(" + y + ")" for x, y in zip(a, b)] for a, b in zip(topk_txt, topk_scores)
    ]
    print(f"Top 5 answers and scores: {topk_all[0]}")
