import numpy as np  # type:ignore
import pandas as pd  # type:ignore
from pathlib import Path
from typing import Union
#import kaggle
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import os

format_md = "* {time:YYYY-MM-DD at HH:mm:ss} | elapsed:{elapsed} | {level} | module:{module} | {name}:{function}:{line} | {message}"
format_stderr = "* <green>{time:YYYY-MM-DD at HH:mm:ss}</green>| elapsed:{elapsed} | module:{module} | <red>{level}</red> | <cyan>{name}:{function}:{line}</cyan> | {message}"


def amex_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting by describing prediction values
    indices = np.argsort(y_pred)[::-1]
    preds, target = y_pred[indices], y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted Gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted Gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted Gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)


def lgb_amex_metric(y_true: np.ndarray, y_pred: np.ndarray):
    """The competition metric with lightgbm's calling convention"""
    return ("amex", amex_metric(y_true, y_pred), True)


# def download_kaggle_competition_data(
#     *,
#     competition_name: str,
#     savepath: Union[str, Path] = ".",
#     overwrite: bool = False,
# ):
#     # Fetch the user's competition list and verigy that supplied value of competition_name is valid
#     api = KaggleApi()
#     api.authenticate()
#     # os.environ['KAGGLE_CONFIG_DIR'] = "/config/.kaggle/"
#     api.competition_download_files(competition_name, savepath)

#     with zipfile.ZipFile(Path(savepath) / competition_name + ".zip", "r") as zipref:
#         zipref.extractall(savepath)

# def download_kaggle_competition_data_file(
#     *,
#     competition_name: str,
#     filename :str,
#     savepath: Union[str, Path] = ".",
#     overwrite: bool = False,
# ):
#     # Fetch the user's competition list and verigy that supplied value of competition_name is valid
#     api = KaggleApi()
#     api.authenticate()
#     # os.environ['KAGGLE_CONFIG_DIR'] = "/config/.kaggle/"
#     api.competition_download_file(competition = competition_name,file_name = filename,path=savepath)
#     filename = filename + ".zip"
#     with zipfile.ZipFile(Path(savepath) / filename, "r") as zipref:
#         zipref.extractall(savepath)

# def download_kaggle_dataset(
#     *,
#     dataset_name: str,
#     savepath: Union[str, Path] = ".",
#     overwrite: bool = False,
# ):

#     api = KaggleApi()
#     api.authenticate()
#     api.dataset_download_files(dataset_name, path=savepath, unzip=True)


# def submit_to_competition(
#     *, competition_name: str, submission_path: Union[str, Path], message: str
# ):
#     api = KaggleApi()
#     api.authenticate()
#     api.competition_submit(
#         file_name=submission_path,
#         competition=competition_name,
#         message=str,
#     )
