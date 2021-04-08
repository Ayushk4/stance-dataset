import argparse

parser = argparse.ArgumentParser()

dataset_path = "glove/indexed.json"
glove_embed = "glove/embed_glove.json"
glove_dims_ = 200

parser.add_argument("--seed", type=int, default=4214)
parser.add_argument("--target_merger", type=str, required=True)
parser.add_argument("--test_mode", type=str, default="True")
parser.add_argument("--cross_valid_num", type=int, default=4, help="For 5-fold crossvalidation, which part is valid set.")

parser.add_argument("--dataset_path", type=str, default=dataset_path)
parser.add_argument("--glove_embed", type=str, default=glove_embed)

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--n_epochs", type=int, default=25)

parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--glove_dims", type=int, default=glove_dims_, help="Dimensions of glove twitter embeddings.")

parser.add_argument("--dummy_run", dest="dummy_run", action="store_true", help="To make the model run on only one training sample for debugging")
parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")

parser.add_argument("--run", type=str, default=None)
parser.add_argument("--wandb",  dest="wandb", action="store_true", default=False)
parser.add_argument("--notarget",  dest="notarget", action="store_true", default=False)

params = parser.parse_args()

assert params.target_merger in ['CVS_AET', 'ANTM_CI', 'AET_HUM', 'CI_ESRX', 'DIS_FOX']
assert params.cross_valid_num >= 0 and params.cross_valid_num <=4
