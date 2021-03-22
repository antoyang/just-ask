# Just Ask: Learning to Answer Questions from Millions of Narrated Videos

This codebase includes data downloading instructions, data preprocessing scripts, training scripts, and evaluation scripts for our paper. The iVQA and HowToVQA69M datasets, pre-extracted video features used for our experiments, and pretrained checkpoints, will also be made publically available soon.

## Paths and Requirements
Fill the empty paths in the file `global_parameters.py`.

To install requirements, run:
```
pip install -r requirements.txt
```

## Downloading Data
**iVQA**: The dataset will be released soon.
The iVQA folder should contain `train.csv`, `val.csv` and `test.csv`.

**MSRVTT-QA**: Download it from https://github.com/xudejing/video-question-answering.
The MSRVTT-QA folder should contain `train_qa.json`, `val_qa.json`, `test_qa.json`, and also `train_val_videodatainfo.json` and `test_videodatainfo.json`. 
The two last files are from the MSR-VTT dataset http://ms-multimedia-challenge.com/2016/dataset, and are used to filter out video IDs in HowTo100M that are in the validation and test sets of MSRVTT-QA.

**MSVD-QA**: Download it from https://github.com/xudejing/video-question-answering. 
The MSVD-QA folder should contain `train_qa.json`, `val_qa.json`, `test_qa.json` and `youtube_mapping.txt`. 
The last file is used to filter out videos IDs in HowTo100M that are in the validation and test sets of MSVD-QA.

**ActivityNet-QA**: Download it from https://github.com/MILVLG/activitynet-qa.
The ActivityNet-QA folder should contain `train_q.json`, `train_a.json`, `val_q.json`, `val_a.json`, `test_q.json` and `test_a.json`.

**How2QA**: Download it from https://github.com/ych133/How2R-and-How2QA.
The How2QA folder should contain `how2QA_train_release.csv` and `how2QA_val_release.csv`.

**HowTo100M**: Download it from https://github.com/antoine77340/howto100m.
The HowTo100M folder should contain `caption_howto100m_with_stopwords.pkl` and `s3d_features.csv`.

Each of these folders should be in `DEFAULT_DATASET_DIR`, and should also contain a `video` subfolder containing the videos downloaded from each dataset.

## Data Preprocessing
**VideoQA**: To prepare and save train, validation and test dataframe files (`train.csv`, `val.csv`, `test.csv`), and the vocabulary map (`vocab.json`) in the open-ended setting, for each VideoQA dataset, use:
```
python preproc/preproc_ivqa.py
python preproc/preproc_msrvttqa.py
python preproc/preproc_msvdqa.py
python preproc/preproc_activitynetqa.py
python preproc/preproc_how2qa.py
```

**HowTo100M**: To preprocess HowTo100M by removing potential intersection with the validation and test sets of VideoQA datasets, and removing repetition in the ASR data, use (this will save `caption_howto100m_sw_nointersec.pickle` and `s3d_features_nointersec.csv` in `HOWTO_PATH`):
```
python preproc/howto100m_remove_intersec.py
python preproc/howto100m_remove_repet.py
```

## Extract video features
We provide in the `extract` folder the code to extract features with the S3D feature extractor. It requires downloading the S3D model weights trained with no manual data annotation available at https://github.com/antoine77340/S3D_HowTo100M. The `s3d_howto100m.pth` checkpoint and `s3d_dict.npy` dictionary should be in `DEFAULT_MODEL_DIR`.

**Extraction**: You should prepare for each dataset a csv with columns `video_path` (typically in the form of *<dataset_path>/video/<video_path>*), and `feature_path` (typically in the form of *<dataset_path>/features/<video_path>.npy*). Then use (you may launch this script on multiple GPUs to fasten the extraction process):
```
python extract/extract.py --csv <csv_path>
```

**Merging**: To merge the extracted features into a single pth file for each VideoQA dataset, use (for ActivityNet-QA that contains long videos, add `--pad 120`):
```
python extract/merge_features.py --folder <dataset_path>/features --output_path <dataset>/s3d.pth --dataset <dataset>
```

For HowTo100M, the features should be stored in `SSD_PATH/s3d/`, one file per video. `SSD_PATH` should preferably on a SSD disk for optimized on-the-fly reading operation time during pretraining.

## HowToVQA69M Generation
This requires downloading the pretrained BRNN model weights from https://github.com/ottokart/punctuator2. The `INTERSPEECH-T-BRNN.pcl` file should be in `DEFAULT_MODEL_DIR`. 

**Punctuating**: First, we punctuate the speech data at the video level and split the video into clips temporally aligned with infered sentences (you may launch this script on multiple CPUs to fasten the process):
```
python videoqa_generation/punctuate.py
```

**Merging infered speech sentences**: Second, we merge the punctuated data into one file:
```
python videoqa_generation/merge_punctuations.py
```

**Extracting answers**: Third, we extract answers from speech transcripts. This requires having cloned https://github.com/patil-suraj/question_generation in `QG_REPO_DIR`. Then use (you may launch this script on multiple GPUs to fasten the process):
```
python videoqa_generation/extract_answers.py
```

**Merging extracted answers**: Fourth, we merge the extracted answers into one file:
```
python videoqa_generation/merge_answers.py
```

**Generating questions**: Fifth, we generate questions pairs from speech and extracted answers. This requires having cloned https://github.com/patil-suraj/question_generation in `QG_REPO_DIR`. Then use (you may launch this script on multiple GPUs to fasten the process):
```
python videoqa_generation/generate_questions.py
```

**Merging generated question-answer pairs**: Finally, we merge the generated question-answer pairs into one file (this will save `sqa.pickle`, `train_sqa.csv` and `val_sqa.csv`):
```
python videoqa_generation/merge_qas.py
```

## Pretraining
DistilBERT tokenizer and model checkpoints will be automatically downloaded from https://huggingface.co/transformers/ in `DEFAULT_MODEL_DIR/transformers`.

**Training VQA-T on HowToVQA69M**:
To train on HowToVQA69M as proposed in our paper (it takes less than 48H on 8 NVIDIA Tesla V100), run:
```
python main_sqa.py --dataset="sqa" --num_thread_reader=16 --epochs=10 --checkpoint_dir="ptsqa" --qmax_words=20 --amax_words=10 --max_feats=20 --batch_size=128 --batch_size_val=256 --n_pair=32 --freq_display=10 --lr=0.00005 --mlm_prob=0.15
```

**Training QA-T on HowToVQA69M**: The pretraining is done with the previous command complemented with `--baseline qa`.

**Training VQA-T on HowTo100M**: To train the multi-modal transformer on HowTo100M with masked language modeling and cross-modal matching objectives (it takes less than 2 days on 8 NVIDIA Tesla V100), run:
```
python main_mlmcm.py --dataset="howto100m" --num_thread_reader=16 --epochs=10 --min_words=10 --qmax_words=20 --min_time=10 --max_feats=20 --checkpoint_dir="pthtm" --batch_size=128 --batch_size_val=3500 --n_pair=32 --freq_display=10 --lr=0.00005 --lr_decay=0.9 --n_negs=1
```
Note that zero-shot validation on YouCook2 and MSR-VTT video retrieval are performed at every epoch. We followed https://github.com/antoine77340/MIL-NCE_HowTo100M for the preprocessing of these datasets.

## VideoQA Finetuning and Zero-Shot VideoQA
**Finetuning**: To finetune on a downstream VideoQA dataset (for MSRVTT-QA, which is the largest downstream dataset, it takes less than 4 hours on 4 NVIDIA Tesla V100), run:
```
python main_videoqa.py --num_thread_reader=16 --checkpoint_dir=<dataset> --mlm_prob=0.15 --dataset=<dataset> --lr=$lr --pretrain_path=<DEFAULT_CKPT_PREDIR>/ptsqa/e9.pth --batch_size_val=2048 --batch_size=256
```

**Training from scratch**: VQA-T trained from scratch is simply obtained by running the previous script with no `pretrain_path` set.

**Zero-shot**: Zero-shot VideoQA can simply be performed by adding `--test 1`. In the case of QA-T, also add `--baseline qa`. In the case of VQA-T pretrained on HowTo100M, run:
```
python eval_videoqa_cm.py --num_thread_reader=16 --checkpoint_dir=<dataset> --dataset=<dataset> --pretrain_path=<DEFAULT_CKPT_PREDIR>/pthtm/e9.pth
```

## Evaluation per question type, answer quartile, and VideoQA Demo
**Detailed evaluation**: Using a trained checkpoint, to perform evaluation segmented per question type and answer quartile, use:
```
python eval_videoqa.py --dataset <dataset> --pretrain_path <DEFAULT_CKPT_PREDIR>/<dataset>/best_model.pth
```

**VideoQA Demo**: Using a trained checkpoint, you can also run a VideoQA example with a video file of your choice, and the question of your choice. For that, use (the dataset indicated here is only used for the definition of the answer vocabulary):
```
python demo_videoqa.py --dataset <dataset> --pretrain_path <DEFAULT_CKPT_PREDIR>/<dataset>/best_model.pth --question_example <question> --video_example <video_path>
```

## Acknowledgements
The video feature extraction code is inspired by https://github.com/antoine77340/video_feature_extractor. The model implementation of our multi-modal transformer (as well as the masked language modeling setup) is inspired by https://huggingface.co/transformers/model_doc/distilbert.html. 
