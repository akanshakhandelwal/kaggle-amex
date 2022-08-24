from mlglance.trainer import AmexTrainer
from mlglance.training_config import TrainingConfig
from mlglance.optuna_params import get_metaclassifier_params
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


c = TrainingConfig(
    data_path="data/",  # denoised data from raddar
    labels_path="data/amex-default-prediction/train_labels.csv",  # original data
    train_feat_path="data/train_1_perc.parquet",
    test_feat_path="data/train_10_perc.parquet",  # aggregated features
    n_splits=2,
    log_path="lgbm_log.md",
    submission_path="submission",
    n_trials=3,
    model=LogisticRegression,
    hyperparams=get_metaclassifier_params,
)

t=AmexTrainer(config=c).setup().stacked_model().create_submission()
