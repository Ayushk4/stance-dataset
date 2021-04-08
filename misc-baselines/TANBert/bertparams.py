import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=4214)
parser.add_argument("--target_merger", type=str, default="Please Enter Test Merger in args",
                    help="Test Merger in 'CVS_AET', 'ANTM_CI', 'AET_HUM', 'CI_ESRX', 'DIS_FOX'"
                )
parser.add_argument("--test_mode", type=str, default="True")
parser.add_argument("--cross_valid_num", type=int, default=4,
                    help="For 5-fold crossvalidation, which part is valid set."
                )

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--n_epochs", type=int, default=5)

parser.add_argument("--dummy_run", dest="dummy_run", action="store_true", help="To make the model run on only one training sample for debugging")
parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")

parser.add_argument("--wandb", dest="wandb", action="store_true", default=False)
parser.add_argument("--bert_type", type=str, required=True)
parser.add_argument("--notarget", dest="notarget", action="store_true", default=False)
parser.add_argument("--dataset_path", type=str, required=True)
params = parser.parse_args()

print("++++++++++++++++++++++")
print("Target =", not params.notarget)
print("++++++++++++++++++++++")
# Change target name for company for wtwt dataset
assert params.dataset_path in ["dataset/data_new.json", "scripts/data_new.json",
                               "scripts/data_1.json", "scripts/data_2.json"
                            ]

assert params.target_merger in ['CVS_AET', 'ANTM_CI', 'AET_HUM', 'CI_ESRX', 'DIS_FOX']
assert params.cross_valid_num >= 0 and params.cross_valid_num <=4
params.test_mode = params.test_mode.lower() == "true"
