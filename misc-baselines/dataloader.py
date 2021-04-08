import torch
import os, json, random, sys
from params import params
import numpy as np

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

basepath = "/".join(os.path.realpath(__file__).split('/')[:-1])
DATA_PATH = os.path.join(basepath, params.dataset_path)


class StanceDataset:
    def __init__(self):
        self.stance2id = {'comment': 0, 'unrelated': 1, 'support': 2, 'refute': 3}

        self.id2stance = {v: k for k,v in self.stance2id.items()}
        print(self.stance2id, "||", self.id2stance)

        train, eval_set = self.load_dataset(DATA_PATH)

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
        self.pad_idx = full_dataset['pad_idx']
        full_dataset = full_dataset['dataset']

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

        idx = 0
        num_data = len(unbatched)
        while idx < num_data:
            texts = []
            targets = []
            stances = []

            for single_tweet in unbatched[idx:min(idx+params.batch_size, num_data)]:
                texts.append(single_tweet["text"])
                targets.append(single_tweet['target'])
                stances.append(self.stance2id[single_tweet["stance"]])

            this_text_maxlen = max(len(x) for x in texts)
            this_target_maxlen = max(len(x) for x in targets)

            if idx + params.batch_size > num_data:
                print(texts, targets)

            get_masks = lambda x, maxlen: [[False] * len(y) + [True] * (maxlen - len(y)) for y in x]            
            pad_mask_text = torch.BoolTensor(get_masks(texts, this_text_maxlen)).to(params.device)
            pad_mask_target = torch.BoolTensor(get_masks(targets, this_target_maxlen)).to(params.device)

            pad_to_max_length = lambda x, maxlen: [y + [self.pad_idx] * (maxlen - len(y)) for y in x]

            texts = torch.LongTensor(pad_to_max_length(texts, this_text_maxlen)).to(params.device)
            targets = torch.LongTensor(pad_to_max_length(targets, this_target_maxlen)).to(params.device)
            stances = torch.LongTensor(stances).to(params.device)

            b = params.batch_size if (idx + params.batch_size) < num_data else (num_data - idx)

            assert texts.size() == torch.Size([b, this_text_maxlen])
            assert targets.size() == torch.Size([b, this_target_maxlen])
            assert stances.size() == torch.Size([b])
            assert pad_mask_text.size() == torch.Size([b, this_text_maxlen])
            assert pad_mask_target.size() == torch.Size([b, this_target_maxlen])
            
            dataset.append((texts, targets, stances, pad_mask_text, pad_mask_target))
            idx += params.batch_size

        print("num_batches=", len(dataset), " | num_data=", num_data)

        return dataset, None
    

if __name__ == "__main__":
    dataset = StanceDataset()
    print("Train_dataset Size =", len(dataset.train_dataset),
            "Eval_dataset Size =", len(dataset.eval_dataset))
    print(len(dataset.train_dataset))#[0])
    print(dataset.train_dataset[-1])
    #print(len(dataset.hard_dataset))
    import os
    os.system("nvidia-smi")
