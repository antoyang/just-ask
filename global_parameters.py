import os

# Fill the paths
DEFAULT_DATASET_DIR = "/gpfsstore/rech/msk/urt22vb/cvpr21/datasets/" # where the datasets folders are
# DEFAULT_DATASET_DIR = "/sequoia/data2/ayang/cvpr21/datasets/" # where the datasets folders are
# DEFAULT_DATASET_DIR = "/home/ROCQ/willow/ayang/data/" # where the pretrained models are
DEFAULT_CKPT_DIR = "/gpfsstore/rech/msk/urt22vb/cvpr21/checkpoints/"  # where the training checkpoints and logs will be saved
DEFAULT_MODEL_DIR = "/gpfsstore/rech/msk/urt22vb/cvpr21/models/"  # where the pretrained models are
# DEFAULT_MODEL_DIR = "/sequoia/data2/ayang/cvpr21/models/"  # where the pretrained models are
# DEFAULT_MODEL_DIR = "/home/ROCQ/willow/ayang/models/" # where the pretrained models are
SSD_DIR = "/gpfsscratch/rech/msk/urt22vb/"  # where the HowTo100M S3D features are

# Map from dataset name to folder name
dataset2folder = {
    "ivqa": "iVQA",
    "msrvtt": "MSRVTT-QA",
    "msvd": "MSVD-QA",
    "activitynet": "ActivityNet-QA",
    "howto100m": "HowTo100M",
    "sqa": "HowTo100M",
    "how2qa": "How2QA"
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
ACT_FEATURES_PATH = os.path.join(
    ACT_PATH, "s3d.pth"
)  # Path to ActivityNet features file
HOWTO_PATH = os.path.join(
    DEFAULT_DATASET_DIR, dataset2folder["howto100m"]
)  # Path where HowTo100M is downloaded
HOW2QA_PATH = os.path.join(
    DEFAULT_DATASET_DIR, dataset2folder["how2qa"]
)  # Path where How2QA is downloaded

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
TRANSFORMERS_PATH = "/gpfswork/rech/msk/urt22vb/transformers"
    # os.path.join(
    # DEFAULT_MODEL_DIR, "transformers"
# )  # Path where the transformers checkpoints will be saved

# Question-answer Generation
punct_dir = os.path.join(SSD_DIR, "punct")  # Path where the punctuated clips will be created (1 file per unique video)
QG_REPO_DIR = "/gpfswork/rech/msk/urt22vb"  # Path where the question generation repo is cloned
answers_dir = os.path.join(SSD_DIR, "ans")  # Path where the extracted answers will be saved (1 file per unique video)
qas_dir = os.path.join(SSD_DIR, "qas") # Path where the generated question-answers will be saved (1 file per unique video)

SERVER_HTML_PATH = "/home/ROCQ/willow/ayang/videoqa/vqa_server.html"
    #"/sequoia/data1/ayang/cvpr21/server_videoqa.html"
SERVER_FEATURE_PATH = "/home/ROCQ/willow/ayang/data"
    #"/sequoia/data2/ayang/cvpr21/datasets/"
