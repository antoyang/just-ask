import os

# Fill the paths
DEFAULT_DATASET_DIR = ""  # where the datasets folders are
DEFAULT_CKPT_DIR = ""   # where the training checkpoints and logs will be saved
DEFAULT_MODEL_DIR = ""  # where the pretrained models are
SSD_DIR = ""  # where the HowTo100M S3D features are
HOWTO_FEATURES_PATH = os.path.join(SSD_DIR, "s3d_features", "howto100m_s3d_features")
WEBVID_FEATURES_PATH = os.path.join(SSD_DIR, "webvid_s3d_features")

# Map from dataset name to folder name
dataset2folder = {
    "ivqa": "iVQA",
    "msrvtt": "MSRVTT-QA",
    "msvd": "MSVD-QA",
    "activitynet": "ActivityNet-QA",
    "howto100m": "HowTo100M",
    "howtovqa": "HowToVQA69M",
    "how2qa": "How2QA",
    "webvidvqa": "WebVidVQA"
}

# Datasets
IVQA_PATH = os.path.join(
    DEFAULT_DATASET_DIR, dataset2folder["ivqa"]
)  # Path where iVQA is downloaded
MSRVTT_PATH = os.path.join(
    DEFAULT_DATASET_DIR, dataset2folder["msrvtt"]
)  # Path where MSRVTT-QA is downloaded
MSVD_PATH = os.path.join(
    DEFAULT_DATASET_DIR, dataset2folder["msvd"]
)  # Path where MSVD-QA is downloaded
ACT_PATH = os.path.join(
    DEFAULT_DATASET_DIR, dataset2folder["activitynet"]
)  # Path where ActivityNet-QA is downloaded
HOWTO_PATH = os.path.join(
    DEFAULT_DATASET_DIR, dataset2folder["howto100m"]
)  # Path where HowTo100M is downloaded
HOWTOVQA_PATH = os.path.join(
    DEFAULT_DATASET_DIR, dataset2folder["howtovqa"]
)  # Path where HowToVQA69M is downloaded / generated
HOW2QA_PATH = os.path.join(
    DEFAULT_DATASET_DIR, dataset2folder["how2qa"]
)  # Path where How2QA is downloaded
WEBVID_PATH = os.path.join(
    DEFAULT_DATASET_DIR, dataset2folder["webvidvqa"]
)  # Path where WebVid is downloaded

# Models
S3D_PATH = os.path.join(
    DEFAULT_MODEL_DIR, "s3d_howto100m.pth"
)  # Path to S3D checkpoint
S3D_DICT_PATH = os.path.join(
    DEFAULT_MODEL_DIR, "s3d_dict.npy"
)  # Path to S3D dictionary
PUNCTUATOR_PATH = os.path.join(
    DEFAULT_MODEL_DIR, "INTERSPEECH-T-BRNN.pcl"
)  # Path to Punctuator2 checkpoint
TRANSFORMERS_PATH = os.path.join(
    DEFAULT_MODEL_DIR, "transformers"
)  # Path where the transformers checkpoints will be saved

# Question-answer Generation
punct_dir = os.path.join(
    SSD_DIR, "punct"
)  # Path where the punctuated clips will be created (1 file per unique video)
QG_REPO_DIR = ""  # Path where the question generation repo is cloned
answers_dir = os.path.join(
    SSD_DIR, "ans"
)  # Path where the extracted answers will be saved (1 file per unique video)
qas_dir = os.path.join(
    SSD_DIR, "qas"
)  # Path where the generated question-answers will be saved (1 file per unique video)
