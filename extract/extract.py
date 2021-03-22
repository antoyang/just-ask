import torch as th
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from extract.video_loader import VideoLoader
from torch.utils.data import DataLoader
from extract.s3dg import S3D
from extract.preprocessing import Preprocessing
from extract.random_sequence_shuffler import RandomSequenceSampler
from global_parameters import S3D_PATH

parser = argparse.ArgumentParser(description="Easy video feature extractor")

parser.add_argument(
    "--csv",
    type=str,
    help="input csv with columns video_path (input video) and feature_path (output path to feature)",
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="batch size for extraction"
)
parser.add_argument(
    "--half_precision",
    type=int,
    default=0,
    help="whether to output half precision float or not",
)
parser.add_argument(
    "--num_decoding_thread",
    type=int,
    default=0,
    help="number of parallel threads for video decoding",
)
parser.add_argument(
    "--l2_normalize",
    type=int,
    default=0,
    help="whether to l2 normalize the output feature",
)
parser.add_argument("--fps", type=int, default=16, help="framerate")
parser.add_argument(
    "--model_path", type=str, default=S3D_PATH, help="path to s3d model checkpoint"
)
parser.add_argument("--mixed_5c", type=int, default=1, help="mixed_5c feature")
parser.add_argument(
    "--feature_dim", type=int, default=1024, help="output video feature dimension"
)
parser.add_argument("--cudnn_benchmark", type=int, default=0, help="cudnn benchmark")
args = parser.parse_args()

if args.cudnn_benchmark:
    th.backends.cudnn.benchmark = True

dataset = VideoLoader(
    args.csv,
    framerate=args.fps,
    size=224,
    centercrop=True,
)
n_dataset = len(dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread,
    sampler=sampler if n_dataset > 10 else None,
)
preprocess = Preprocessing(num_frames=args.fps)
model = S3D(512, space_to_depth=True, embd=args.mixed_5c, feature_map=0)
model_data = th.load(args.model_path)
model.load_state_dict(model_data)
model.eval()
model = th.nn.DataParallel(model)
model = model.cuda()

with th.no_grad():
    for k, data in enumerate(loader):
        input_file = data["input"][0]
        output_file = data["output"][0]
        if len(data["video"].shape) > 3:
            print(
                "Computing features of video {}/{}: {}".format(
                    k + 1, n_dataset, input_file
                )
            )
            video = data["video"].squeeze()
            if len(video.shape) == 4:
                video = preprocess(video)
                n_chunk = len(video)
                features = th.cuda.FloatTensor(n_chunk, args.feature_dim).fill_(0)
                n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                for i in tqdm(range(n_iter)):
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    video_batch = video[min_ind:max_ind].cuda()
                    batch_features = model(video_batch)
                    if args.l2_normalize:
                        batch_features = F.normalize(batch_features, dim=1)
                    features[min_ind:max_ind] = batch_features
                features = features.cpu().numpy()
                if args.half_precision:
                    features = features.astype("float16")
                np.save(output_file, features)
        else:
            print("Video {} already processed.".format(input_file))
