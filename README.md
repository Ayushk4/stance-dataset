# Stance Dataset - </i>t</i>WTWT

<i>t</i>WT–WT: A Dataset to Assert the Role of Target Entities for Detecting Stance of Tweets

**Accepted to appear** at NAACL-HLT 2021.

This repository contains the dataset and baseline accompanying the paper.

ArXiv link: `coming-soon`

## Overview

### Abstract

The stance detection task aims at detecting the stance of a tweet or a text for a target. These targets can be named entities or free-form sentences (claims). Though the task involves reasoning of the tweet with respect to a target, we find that it is possible to achieve high accuracy on several publicly available Twitter stance detection datasets without looking at the target sentence. Specifically, a simple tweet classification model achieved human-level performance on the WT–WT dataset and more than two-third accuracy on various other datasets. We investigate the existence of biases in such datasets to find the potential spurious correlations of sentiment-stance relations and lexical choice associated with the stance category. Furthermore, we propose a new large dataset free of such biases and demonstrate its aptness on the existing stance detection systems. Our empirical findings show much scope for research on the stance detection task and proposes several considerations for creating future stance detection datasets.

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
- `dataset`: Containing the <i>t</i>WT–WT datasets
  - `dataset/process.py`: Process the dataset
  - `dataset/wtwt_new.json`: Process the dataset
- `scripts`: To re-create the <i>t</i>WT–WT dataset from WT–WT
  - `dataset/generate.py`: Generate the <i>t</i>WT–WT dataset from WT–WT dataset
  - `dataset/process.py`: Process the dataset
  - `dataset/wtwt_new.json`: The<i>t</i>WT–WT dataset.
- `misc-baselines`: Other baselines
  - `dataloader.py`: Loader for LSTM+Glove based baselines.
  - `index_dataset.py`: Indexes the dataset for LSTM based baselines.
  - `params.py`: Argparsing to enable easy experiments.
  - `prepare_glove.py`: Prepares the Glove Matrix based on vocab.
  - `smaller_glove.py`: Filters the words common to Glove and dataset.
  - `siam-net.py`: The original SiamNet model and training functions.
  - `BertSiamNet`: Contains SiamNet with Bert features.
    - `bert_siam-net.py`: The SiamNet model with Bert feats model and training functions.
    - `bertloader.py`: Dataloader for SiamNet
    - `bertparams.py`: Argparsing to enable easy experiments.
  - `TANBert`: Contains TAN model with Bert features.
    - `bert_tan.py`: The TAN model with Bert feats and training functions.
    - `bertloader.py`: Dataloader for TAN
    - `bertparams.py`: Argparsing to enable easy experiments.


### 1. Setting up the codebase and the dependencies.

- Clone this repository - `git clone https://github.com/Ayushk4/stance-dataset`
- Follow the instructions from the [`Dependencies`](#dependencies) Section above to install the dependencies.
- If you are interested in logging your runs, Set up your wandb - `wandb login`.

### 2. Setting up the datasets.

This codebase supports the <i>t</i>WT–WT dataset considered in our paper.

- Please extract the tweets for the ids in `dataset/wtwt_new.json`. To do this, please register your application on the twitter developer API and download the tweets. Save all the tweets in a single folder at the path `dataset/scrapped_full/` with each file named in the format <tweet_id>.json where tweet_id is a 17-20 digit tweet id.

- Add the desired target sentences for each merger in `dataset/merger2target.json` inside this folder. Similar to the WT–WT dataset, we leave it to the user of the dataset to experiments with different target sentences for each merger.

- The process the dataset by `cd dataset` and then `python3 process.py`. The final processed data will stored in a json format at `dataset/data_new.json`, which will be input to our baseline.

### 3. Training the models.

This codebase supports the four models we experimented in the paper.

- Target Oblivious Bert

<img src="https://github.com/Ayushk4/stance-dataset/blob/master/images/target-oblivious-bert.png" alt="target-oblivious-bert" width="200"/>

- Target Aware Bert

<img src="https://github.com/Ayushk4/stance-dataset/blob/master/images/target-aware-bert.png" alt="target-aware-bert" width="200"/>

- Bert Siamese Network

<img src="https://github.com/Ayushk4/stance-dataset/blob/master/images/siamese-net-bert.png" alt="siamese-net-bert" width="300"/>

- TAN with Bert

<img src="https://github.com/Ayushk4/stance-dataset/blob/master/images/tan-bert.png" alt="tan-bert" width="300"/>



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

If you want to test out our other baselines mentioned in the paper, then first `cd misc-baselines/BertSiamNet` or `cd misc-baselines/TANBert` and execute `python3 bert_siam-net.py [ARGS]` or `python3 bert_tan.py [ARGS]` with same set of args as above.

If you want to execute lstm based baselines, then `cd misc-baselines` and download [glove twitter embeddings](https://nlp.stanford.edu/projects/glove/) inside `misc-baselines/glove`. Execute `python3 smaller_glove.py`, `python3 prepare_glove.py` and `python3 index_dataset.py` with the arguments `--glove_dims` and `--min_occur` denoting the desired glove dimensions and minimum occurence of a token to be added to the vocabulary. Then execute `python3 {model_name}.py [ARGS]`, where `[ARGS]` are same as above with additional argument of `--glove_dims`.


### Generating <i>t</i>WT–WT dataset (not recommended)

To replicate steps to generate <i>t</i>WT–WT dataset from WT–WT dataset please run `python3 scripts/generate.py`. We are releasing this script, but do not recommended to be used. Since the baselines have been evaluation on the released <i>t</i>WT–WT dataset in `dataset/wtwt_new.json`, and it will serve as a fairer common new leaderboard for stance detection task.

The above step can also be performed after copying (or symlinking) the `scrapped_full` folder from step 2 above to inside the scripts folder along with copying the `wtwt_ids.json` [file](https://github.com/cambridge-wtwt/acl2020-wtwt-tweets) to inside the `scripts` folder.

The above `scripts/generate.py` will generate the <i>t</i>WT–WT dataset along with the two intermediate datasets after first and second steps which can be used for ablation study. All three of these can be (pre-)processed using `python3 scripts/process.py` which can be used by the model through specifying the respective dataset_path.


## Results


| Model            | CVS_AET F1 | CI_ESRX F1 | ANTM_CI F1 | AET_HUM F1 | Average F1 | Weighted F1 | DIS_FOX F1 |
| ---------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ----------- | ---------- |
| Bert (no-target) | 0.161      | 0.258      | 0.297      | 0.340      | 0.264      | 0.260       | 0.163      |
| Bert (target)    | 0.460      | 0.386      | 0.596      | 0.598      | 0.510      | 0.527       | 0.365      |
| Siam Net         | 0.293      | 0.292      | 0.273      | 0.398      | 0.312      | 0.310       | 0.150      |
| TAN              | 0.170      | 0.222      | 0.308      | 0.332      | 0.258      | 0.260       | 0.150      |
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
          title={tWT–WT: A Dataset to Assert the Role of Target Entities for Detecting Stance of Tweets},
          author={Kaushal, Ayush and Saha, Avirup and Ganguly, Niloy} 
          booktitle={Proceedings of the 2021 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2021)},
          year={2021}
        }
```



## Miscellanous

- You may contact us by opening an issue on this repo and/or mailing to the first author - `<this_github_username> [at] gmail.com` Please allow 2-3 days of time to address the issue.

- The codebase has been written from scratch, but was inspired from many others [1](https://github.com/jackroos/VL-BERT) [2](https://propaganda.qcri.org/fine-grained-propaganda-emnlp.html) [3](https://github.com/prajwal1210/Stance-Detection-in-Web-and-Social-Media)

- License: MIT

