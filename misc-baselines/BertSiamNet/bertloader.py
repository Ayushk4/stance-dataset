import torch
import os, json, random, sys
from bertparams import params
import numpy as np
from transformers import AutoTokenizer

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

MAX_LEN = 0

basepath = "/".join(os.path.realpath(__file__).split('/')[:-1]) + '/../../'
DATA_PATH = os.path.join(basepath, params.dataset_path)


class StanceDataset:
    def __init__(self):
        self.stance2id = {'comment': 0, 'unrelated': 1, 'support': 2, 'refute': 3}

        self.id2stance = {v: k for k,v in self.stance2id.items()}
        print(self.stance2id, "||", self.id2stance)

        train, eval_set = self.load_dataset(DATA_PATH)

        if params.bert_type == "vinai/bertweet-base":
            self.bert_tokenizer = AutoTokenizer.from_pretrained(params.bert_type, normalization=True)
        else:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(params.bert_type)
        new_special_tokens_dict = {"additional_special_tokens": ["<number>", "<money>", "<user>"]}
        self.bert_tokenizer.add_special_tokens(new_special_tokens_dict)
        print("Loaded Bert Tokenizer")

        if params.dummy_run == True:
            self.train_dataset, self.criterion_weights = self.batched_dataset([train[0]] * 2)
            self.eval_dataset, _ = self.batched_dataset([train[0]] * 2)
        else:
            print("Train_dataset:", end= " ")
            self.train_dataset, self.criterion_weights = self.batched_dataset(train)
            print("Eval_dataset:", end= " ")
            self.eval_dataset, _ = self.batched_dataset(eval_set)

        # self.criterion_weights = torch.tensor(self.criterion_weights.tolist()).to(params.device)
        # print("Training loss weighing = ", self.criterion_weights)

    def cross_valiation_split(self, train):
        assert params.test_mode != True, "Cross Validation cannot be done while testing"
        split = len(train) // 5
        valid_num = params.cross_valid_num
        if valid_num == 4:
            split *= 4
            train, valid = train[:split], train[split:]
        elif valid_num == 0:
            train, valid = train[split:], train[:split]
        else:
            train, valid = train[:(split * valid_num)] + train[(split * (valid_num+1)):],   train[(split * valid_num):(split * (valid_num+1))]
        return train, valid

    def load_dataset(self, path):
        # Load the dataset
        full_dataset = json.load(open(path, "r"))

        # Split the dataset
        train, valid, test = [], [], []
        for data in full_dataset:
            if data['merger'] == "DIS_FOX":
                # For Dis fox merge this only appears as a test set in all settings.
                if params.target_merger == "DIS_FOX":
                    test.append(data)
            else:
                assert data['merger'] in ['CVS_AET', 'ANTM_CI', 'AET_HUM', 'CI_ESRX']
                if params.target_merger == data['merger']:
                    test.append(data)
                else:
                    train.append(data)

        print("Length of train, valid, test:", len(train), len(valid), len(test))
        if params.test_mode:
            train += valid
            valid = []
            assert len(train) != 0 and len(test) != 0
            eval_set = test
        else:
            assert len(valid) == 0
            train, valid = self.cross_valiation_split(train)
            eval_set = valid
        print("Length of train, eval_set:", len(train), len(eval_set))
        print("Before shuffling train[0] = ", train[0]["tweet_id"], end=" | ")
        random.shuffle(train)
        print("After shuffling train[0] = ", train[0]["tweet_id"])

        return train, eval_set

    def batched_dataset(self, unbatched): # For batching full or a part of dataset.
        dataset = []
        criterion_weights = np.zeros(4) + 0.0000001 # 4 labels 

        idx = 0
        num_data = len(unbatched)

        while idx < num_data:
            batch_texts = []
            batch_targets = []
            stances = []
            
            for single_tweet in unbatched[idx:min(idx+params.batch_size, num_data)]:
                this_stance_ids = self.stance2id[single_tweet["stance"]]
                criterion_weights[this_stance_ids] += 1
                stances.append(this_stance_ids)

                this_tweet = single_tweet['text']

                if params.notarget:
                    this_target = ""
                else:
                    this_target = single_tweet["target"]
                batch_targets.append([this_target, ''])
                batch_texts.append([this_tweet, ''])

            tokenized_batch_text = self.bert_tokenizer.batch_encode_plus(batch_texts, pad_to_max_length=True,
                                                            return_tensors="pt", return_token_type_ids=True)
            tokenized_batch_target = self.bert_tokenizer.batch_encode_plus(batch_targets,
                                    pad_to_max_length=True, return_tensors="pt", return_token_type_ids=True)

            texts = tokenized_batch_text['input_ids'].to(params.device)
            targets = tokenized_batch_target['input_ids'].to(params.device)
            stances = torch.LongTensor(stances).to(params.device)
            text_pad_masks = tokenized_batch_text['attention_mask'].squeeze(1).to(params.device)
            target_pad_masks = tokenized_batch_target['attention_mask'].squeeze(1).to(params.device)

            global MAX_LEN
            MAX_LEN = max(MAX_LEN, texts.shape[1])
            # print("\n", stances, stances.size())
            # print("\n", pad_masks[0, :], pad_masks.size())
            # print("\n", segment_embed[0, :], segment_embed.size())

            b = params.batch_size if (idx + params.batch_size) < num_data else (num_data - idx)
            l = texts.size(1)
            assert texts.size() == torch.Size([b, l])
            assert text_pad_masks.size() == torch.Size([b, l])
            l = targets.size(1)
            assert targets.size() == torch.Size([b, l])
            assert target_pad_masks.size() == torch.Size([b, l])
            assert stances.size() == torch.Size([b])

            dataset.append((texts, targets, stances, text_pad_masks, target_pad_masks))
            idx += params.batch_size

        print("num_batches=", len(dataset), " | num_data=", num_data)
        criterion_weights = np.sum(criterion_weights)/criterion_weights
        print(MAX_LEN)
        return dataset, criterion_weights/np.sum(criterion_weights)

if __name__ == "__main__":
    dataset = StanceDataset()
    print("Train_dataset Size =", len(dataset.train_dataset),
            "Eval_dataset Size =", len(dataset.eval_dataset))
    print(len(dataset.train_dataset))#[0])
    print(dataset.train_dataset[-1])
    #print(len(dataset.hard_dataset))
    import os
    os.system("nvidia-smi")
    print(MAX_LEN)
