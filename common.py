import flair, torch
from os import path
from click import BadParameter

default_seed = 42
default_model = "C"
default_fold = 1
data_folder = "./MIM-GOLD-SETS.21.05/sets"

pretrained_model = "jonfd/convbert-base-igc-is"


def make_deterministic(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    flair.set_seed(seed)


def make_sure_fold_files_exist(data_folder, fold):
    fold_str = str(format(fold, "02"))
    if not path.isfile(path.abspath(f"{data_folder}/{fold_str}TM.tsv")):
        exit(f"Unable to find training data, '{fold_str}TM.tsv' in '{data_folder}'.")
    if not path.isfile(path.abspath(f"{data_folder}/{fold_str}PM.tsv")):
        exit(f"Unable to find test data, '{fold_str}PM.tsv' in '{data_folder}'.")
    print(f"Data folder: '{data_folder}'.")


def validate_fold(ctx, param, value):
    fold = int(value)
    if fold < 1 or fold > 10:
        raise BadParameter("Fold needs to be in the range 1 to 10.")
    return fold


def transform_tag_column(ctx, param, value):
    return int(value)