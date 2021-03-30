# Stance Dataset - WTWTv2

Stance Detection is not Classification: Promoting the role of Target Entities for Detecting Stance.

Accepted to appear at NAACL-HLT 2021.

This repository contains the dataset and baseline accompanying the paper.

## Overview

### Abstract

The stance detection task aims at detecting the stance of a tweet or a text with respect to a target. These targets can be named entities or free-form sentences (claims). Though the task involves reasoning of the tweet with respect to a target, we find that it is possible to achieve high accuracy on several publicly available Twitter stance detection datasets without looking at the target sentence. Specifically, a simple tweet classification model achieved human-level performance on the WT-WT dataset and more than two-third accuracy on a variety of other datasets. We carry out an investigation into the existence of biases in such datasets to find the potential spurious correlations of sentiment-stance relations and lexical choice associated with stance category. Furthermore, we propose a new large dataset free of such biases and demonstrate its aptness on the existing stance detection systems. Our empirical findings much scope for research on stance detection and proposes several considerations for creating future stance detection datasets.

![Alt text](https://github.com/Ayushk4/stance-dataset/blob/master/images/image-stance-dataset.jpg)


## Dependencies

| Dependency                  | Version | Installation Command                                                |
| ----------                  | ------- | ------------------------------------------------------------------- |
| Python                      | 3.8.5   | `conda create --name stance python=3.8` and `conda activate stance` |
| PyTorch, cudatoolkit        | 1.5.0   | `conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch`          |
| Transformers  (HuggingFace) | 3.5.0   | `pip install transformers==3.5.0`     |
| Scikit-learn                | 0.23.1  | `pip install scikit-learn==0.23.1`    |
| scipy                       | 1.5.0   | `pip install scipy==1.5.0`            |
| Ekphrasis                   | 0.5.1   | `pip install ekphrasis==0.5.1`        |
| emoji                       | 0.6.0   | `pip install emoji`                   |
| wandb                       | 0.9.4   | `pip install wandb==0.9.4`            |


## Instructions


### Directory Structure

Following is the structure of the codebase, in case you wish to play around with it.

- `train.py`: Model and training loop.
- `bertloader.py`: Common Dataloader for the 6 datasets.
- `params.py`: Argparsing to enable easy experiments.
- `README.md`: This file :slightly_smiling_face:
- `.gitignore`: N/A
- `dataset`: Containing the WTWTv2 datasets
  - `dataset/process.py`: Process the dataset
  - `dataset/wtwt_new.json`: Process the dataset
- `scripts`: To re-create the WTWTv2 dataset from WTWT
  - `dataset/generate.py`: Generate the WTWTv2 dataset from WTWT dataset
  - `dataset/process.py`: Process the dataset
  - `dataset/wtwt_new.json`: The WTWTv2 dataset.


### 1. Setting up the codebase and the dependencies.

- Clone this repository - `git clone https://github.com/Ayushk4/stance-dataset`
- Follow the instructions from the [`Dependencies`](#dependencies) Section above to install the dependencies.
- If you are interested in logging your runs, Set up your wandb - `wandb login`.

### 2. Setting up the datasets.

This codebase supports the WTWTv2 dataset considered in our paper.

- Please extract the tweets for the ids in `dataset/wtwt_new.json`. To do this, please register your application on the twitter developer API and download the tweets. Save all the tweets in a single folder at the path `dataset/scrapped_full/` with each file named in the format <tweet_id>.json where tweet_id is a 17-20 digit tweet id.

- Add the desired target sentences for each merger in `dataset/merger2target.json` inside this folder. Similar to the WTWT dataset, we leave it to the user of the dataset to experiments with different target sentences for each merger.

- The process the dataset by `cd dataset` and then `python3 process.py`. The final processed data will stored in a json format at `dataset/data_new.json`, which will be input to our baseline.

### 3. Training the models.

After following the above steps, move to the basepath for this repository - `cd stance-dataset` and recreate the experiments by executing `python3 train.py [ARGS]` where `[ARGS]` are the following:

Required Args:
- dataset_path: The path of dataset; Example Usage: `--dataset_path=dataset/data_new.json`; Type: `str`;
- target_merger: The target merger to be tested upon. Example Usage: `--target_merger=CVS_AET`; Type: `str`; Valid Arguments: ['CVS_AET', 'ANTM_CI', 'AET_HUM', 'CI_ESRX', 'DIS_FOX']; This is a required argument.
- test_mode: Indicates whether to evaluate on the test in the run; Example Usage: `--test_mode=False`; Type: `str`
- bert_type: A required argument to specify the bert weights to be loaded. Refer [HuggingFace](https://huggingface.co/models). Example Usage: `--bert_type=bert-base-cased`; Type: `str`

Optional Args:
- seed: The seed for the current run. Example Usage: `--seed=1`; Type: `int`
- cross_validation_num: A helper input for cross validation . Example Usage: `--cross_valid_num=1`; Type: `int`
- batch_size: The batch size. Example Usage: `--batch_size=16`; Type: `int`
- lr: Learning Rate. Example Usage: `--lr=1e-5`; Type: `float`
- n_epochs: Number of epochs. Example Usage: `--n_epochs=5`; Type: `int`
- dummy_run: Include `--dummy_run` flag to perform a dummy run with a single training and validation batch.
- device: CPU or CUDA. Example Usage: `--device=cpu`; Type: `str`
- wandb: Include `--wandb` flag if you want your runs to be logged to wandb.
- notarget: Include `--notarget` flag if you want the model to be target oblivious.

### Steps to replicate generation of WTWTv2 dataset

To replicate steps to generate WTWTv2 dataset from WTWT dataset please run `python3 scripts/generate.py`. This is being released only to reproduce, but not recommended to use, as we intended the released WTWTv2 dataset in `dataset/wtwt_new.json` to serve as the new leaderboard for stance detection task.

The above step can also be performed after copying (or symlinking) the `scrapped_full` folder from step 2 above to inside the scripts folder along with copying the `wtwt_ids.json` [file](https://github.com/cambridge-wtwt/acl2020-wtwt-tweets) to inside the `scripts` folder.

The above `scripts/generate.py` will generate the WTWTv2 dataset along with the two intermediate datasets after first and second steps which can be used for ablation study. All three of these can be (pre-)processed using `python3 scripts/process.py` which can be used by the model through specifying the respective dataset_path.


## Results


| Model            | CVS_AET F1 | CI_ESRX F1 | ANTM_CI F1 | AET_HUM F1 | Average F1 | Weighted F1 | DIS_FOX F1 |
| ---------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ----------- | ---------- |
| Bert (no-target) | 0.161      | 0.258      | 0.297      | 0.340      | 0.264      | 0.260       | 0.163      |
| Bert (target)    | 0.460      | 0.386      | 0.596      | 0.598      | 0.510      | 0.526       | 0.365      |
| Random guessing  | 0.233      | 0.206      | 0.225      | 0.223      | 0.222      | 0.225       | 0.205      |
| Majority guessing| 0.145      | 0.198      | 0.181      | 0.177      | 0.175      | 0.171       | 0.169      |



## Trained Models



| Model               | Accuracy | F1-Wtd   | F1-Macro | Batch | lr   | Epoch | Model Weights |
| ----------------    | -------- | -------- | -------- | ----- | ---- | ----- | ------------- |
| AET_HUM notarget    | 0.507    | 0.464    | 0.340    | 16    | 1e-5 | 5     | [Link](https://github.com/Ayushk4/stance-dataset/releases/download/v0.0.0/AET_HUM_notar.pt)
| AET_HUM target      | 0.623    | 0.620    | 0.598    | 16    | 3e-5 | 5     | [Link](https://github.com/Ayushk4/stance-dataset/releases/download/v0.0.0/AET_HUM_tar.pt)
| ANTM_CI notarget    | 0.521    | 0.459    | 0.297    | 16    | 1e-5 | 5     | [Link](https://github.com/Ayushk4/stance-dataset/releases/download/v0.0.0/ANTM_CI_notar.pt)
| ANTM_CI target      | 0.697    | 0.666    | 0.596    | 16    | 1e-5 | 2     | [Link](https://github.com/Ayushk4/stance-dataset/releases/download/v0.0.0/ANTM_CI_tar.pt)
| CI_ESRX notarget    | 0.444    | 0.443    | 0.258    | 16    | 1e-5 | 2     | [Link](https://github.com/Ayushk4/stance-dataset/releases/download/v0.0.0/CI_ESRX_notar.pt)
| CI_ESRX target      | 0.569    | 0.550    | 0.386    | 16    | 1e-5 | 2     | [Link](https://github.com/Ayushk4/stance-dataset/releases/download/v0.0.0/CI_ESRX_tar.pt)
| CVS_AET notarget    | 0.339    | 0.235    | 0.161    | 16    | 1e-5 | 5     | [Link](https://github.com/Ayushk4/stance-dataset/releases/download/v0.0.0/CVS_AET_notar.pt)
| CVS_AET target      | 0.485    | 0.462    | 0.460    | 16    | 1e-5 | 2     | [Link](https://github.com/Ayushk4/stance-dataset/releases/download/v0.0.0/CVS_AET_tar.pt)
| DIS_FOX notarget    | 0.432    | 0.317    | 0.163    | 16    | 3e-6 | 2     | [Link](https://github.com/Ayushk4/stance-dataset/releases/download/v0.0.0/DIS_FOX_notar.pt)
| DIS_FOX target      | 0.438    | 0.472    | 0.365    | 16    | 1e-6 | 5     | [Link](https://github.com/Ayushk4/stance-dataset/releases/download/v0.0.0/DIS_FOX_tar.pt)


## Citation

- Authors: Ayush Kaushal, Avirup Saha and Niloy Ganguly
- Code base written by Ayush Kaushal
- To appear at NAACL 2021 Proceedings

Please Cite our paper if you find the codebase useful:

```
@inproceedings{kaushal2020stance,
          title={Stance Detection is not Classification: Increasing the Role of Target Entities for Detecting Stance},
          author={Kaushal, Ayush and Saha, Avirup and Ganguly, Niloy} 
          booktitle={Proceedings of the 2021 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2021)},
          year={2021}
        }
```



## Miscellanous

- You may contact us by opening an issue on this repo and/or mailing to the first author - `<this_github_username> [at] gmail.com` Please allow 2-3 days of time to address the issue.

- The codebase has been written from scratch, but was inspired from many others [1](https://github.com/jackroos/VL-BERT) [2](https://propaganda.qcri.org/fine-grained-propaganda-emnlp.html) [3](https://github.com/prajwal1210/Stance-Detection-in-Web-and-Social-Media)

- License: MIT

