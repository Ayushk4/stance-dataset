# Load Packages and setup wandb
from params import params
import wandb
if params.wandb:
    wandb.init(project="Biasstance", name=params.run)
    wandb.config.update(params)

from dataloader import StanceDataset
import json, os, random

import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

def train(model, dataset, criterion):
    model.train()
    train_losses = []
    num_batch = 0

    for batch in dataset:
        (texts, targets, stances, pad_mask_text, pad_mask_target) = batch
        preds = model(texts, targets, pad_mask_text, pad_mask_target)
        loss = criterion(preds, stances)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
      #  scheduler.step()

        if num_batch % 100 == 0:
            print("Train loss at {}:".format(num_batch), loss.item())

        num_batch += 1
        train_losses.append(loss.item())

    return np.average(train_losses)

def evaluate(model, dataset, criterion, target_names):
    model.eval()
    valid_losses = []
    predicts = []
    gnd_truths = []

    with torch.no_grad():
        for batch in dataset:
            (texts, targets, stances, pad_mask_text, pad_mask_target) = batch
            preds = model(texts, targets, pad_mask_text, pad_mask_target)

            loss = criterion(preds, stances)

            predicts.extend(torch.max(preds, axis=1)[1].tolist())
            gnd_truths.extend(stances.tolist())
            valid_losses.append(loss.item())

    assert len(predicts) == len(gnd_truths)

    confuse_mat = confusion_matrix(gnd_truths, predicts)
    if params.dummy_run:
        classify_report = {"hi": {"fake": 1.2}}
    else:
        classify_report = classification_report(gnd_truths, predicts, target_names=target_names, output_dict=True)

    mean_valid_loss = np.average(valid_losses)
    print("Valid_loss", mean_valid_loss)
    print(confuse_mat)

    for labl in target_names:
        print(labl,"F1-score:", classify_report[labl]["f1-score"])
    print("Accu:", classify_report["accuracy"])
    print("F1-Weighted", classify_report["weighted avg"]["f1-score"])
    print("F1-Avg", classify_report["macro avg"]["f1-score"])

    return mean_valid_loss, confuse_mat ,classify_report


########## Load dataset #############
dataset_object = StanceDataset()
train_dataset = dataset_object.train_dataset
eval_dataset = dataset_object.eval_dataset

if params.dummy_run:
    eval_dataset = train_dataset
    target_names = []
else:
    eval_dataset = dataset_object.eval_dataset
    target_names = [dataset_object.id2stance[id_] for id_ in range(0, 4)]


print("Dataset created")
os.system("nvidia-smi")


########## Create model #############

class SiamNet(nn.Module):
    def __init__(self):
        super(SiamNet, self).__init__()

        basepath = "/".join(os.path.realpath(__file__).split('/')[:-1])
        DATA_PATH = os.path.join(basepath, params.dataset_path)

        self.padding_idx = json.load(open(DATA_PATH, "r"))['pad_idx']

        embedding = json.load(open(params.glove_embed))
        self.embedding_layer = nn.Embedding(self.padding_idx + 1, params.glove_dims, padding_idx=self.padding_idx)
        self.embedding_layer.weight.data.copy_(torch.cat([
                                                torch.Tensor(embedding),
                                                torch.zeros(self.padding_idx + 1 - len(embedding), params.glove_dims)
                                            ], 0)
                                        )
        self.embedding_dropout = nn.Dropout(params.dropout)

        self.hidden = (torch.autograd.Variable(torch.zeros(2, 1, params.glove_dims)).to(params.device),
                        torch.autograd.Variable(torch.zeros(2, 1, params.glove_dims)).to(params.device))

        self.lstm = nn.LSTM(params.glove_dims, params.glove_dims, bidirectional=True)
        self.lstm_dropout = nn.Dropout(params.dropout)

        self.u = nn.Parameter(torch.randn(params.glove_dims * 2, 1))
        self.att_score = nn.Sequential(nn.Linear(params.glove_dims * 2, params.glove_dims * 2), nn.Tanh())

        self.distance_metric = lambda o1, o2: torch.exp(-torch.sum((o1 - o2).abs(), 1)).unsqueeze(1)
        self.classifier_mlp = nn.Sequential(
                                    nn.Linear(params.glove_dims * 4 + 1, params.glove_dims),
                                    nn.Dropout(params.dropout),
                                    nn.Linear(params.glove_dims, 4)
                                )


    def forward(self, texts, targets, pad_mask_text, pad_mask_target):
        texts = self.embedding_dropout(self.embedding_layer(texts))
        targets = self.embedding_dropout(self.embedding_layer(targets))

        h, c = (self.hidden[0].expand(-1, texts.shape[0], -1).contiguous(),
                self.hidden[0].expand(-1, texts.shape[0], -1).contiguous())
        text_contextualized = self.lstm_dropout(self.lstm(texts.permute(1, 0, 2), (h ,c))[0].permute(1, 0, 2))

        h, c = (self.hidden[0].expand(-1, targets.shape[0], -1).contiguous(),
                self.hidden[0].expand(-1, targets.shape[0], -1).contiguous())
        target_contextualized = self.lstm_dropout(self.lstm(targets.permute(1, 0, 2), (h ,c))[0].permute(1, 0, 2))

        u_vector = self.u.T.expand(text_contextualized.shape[0], -1).unsqueeze(1)

        ei_text = (u_vector.expand(-1, text_contextualized.shape[1], -1) * self.att_score(text_contextualized)).sum(2)
        scores_text = ei_text.masked_fill(pad_mask_text, -10000.0).softmax(1)
        text_vector = torch.sum(scores_text.unsqueeze(2) * text_contextualized, 1)

        ei_target = (u_vector.expand(-1, target_contextualized.shape[1], -1) * self.att_score(target_contextualized)).sum(2)
        scores_target = ei_target.masked_fill(pad_mask_target, -10000.0).softmax(1)
        target_vector = torch.sum(scores_target.unsqueeze(2) * target_contextualized, 1)

        distance = self.distance_metric(text_vector, target_vector)

        scores = self.classifier_mlp(torch.cat([distance, text_vector, target_vector], -1))
        return scores

model = SiamNet()
print("Model created")
os.system("nvidia-smi")

print(sum(p.numel() for p in model.parameters()))
model = model.to(params.device)
print("Detected", torch.cuda.device_count(), "GPUs!")
# model = torch.nn.DataParallel(model)

if params.wandb:
    wandb.watch(model)

########## Optimizer & Loss ###########

#criterion = torch.nn.CrossEntropyLoss(weight=dataset_object.criterion_weights, reduction='sum')
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)

# valid_loss, confuse_mat, classify_report = evaluate(model, eval_dataset, criterion, target_names)

for epoch in range(params.n_epochs):
    print("\n\n========= Beginning", epoch+1, "epoch ==========")

    train_loss = train(model, train_dataset, criterion)
    if not params.dummy_run:
        print("EVALUATING:")
        valid_loss, confuse_mat, classify_report = evaluate(model, eval_dataset, criterion, target_names)
    else:
        valid_loss = 0.0

    if not params.dummy_run and params.wandb:
        wandb_dict = {}
        for labl in target_names:
            for metric, val in classify_report[labl].items():
                if metric != "support":
                    wandb_dict[labl + "_" + metric] = val

        wandb_dict["F1-Weighted"] = classify_report["weighted avg"]["f1-score"]
        wandb_dict["F1-Avg"] = classify_report["macro avg"]["f1-score"]
        wandb_dict["Accuracy"] = classify_report["accuracy"]
        wandb_dict["Train_loss"] = train_loss
        wandb_dict["Valid_loss"] = valid_loss

        wandb.log(wandb_dict)

    epoch_len = len(str(params.n_epochs))
    print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                    f'train_loss: {train_loss:.5f}' +
                    f'valid_loss: {valid_loss:.5f}')
    print(print_msg)

if params.test_mode:
    basepath = os.path.join("/".join(os.path.realpath(__file__).split('/')[:-1]),
                            "saves")
    folder_name = params.dataset_path.replace('/', '_') + "_" + params.target_merger 
    folder_name = os.path.join(basepath, folder_name)
    print(folder_name)
    if os.path.isdir(folder_name):
        os.system("rm -rf " + folder_name)
    os.mkdir(folder_name)

    # Store params
    json.dump(vars(params), open(os.path.join(folder_name, "params.json"), 'w+'))

    # Save model
    torch.save(model.state_dict(), os.path.join(folder_name, "model.pt"))

    # Store logs (accuracy)
    logs = {"Accu:": classify_report["accuracy"],
            "F1-Weighted": classify_report["weighted avg"]["f1-score"],
            "F1-Avg": classify_report["macro avg"]["f1-score"]
        }
    json.dump(logs, open(os.path.join(folder_name, "logs.json"), 'w+'))

