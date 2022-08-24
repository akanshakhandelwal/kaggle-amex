from mlglance.trainer import AmexTrainer
from mlglance.training_config import TrainingConfig
from mlglance.optuna_params import get_xgb_params
from xgboost import XGBClassifier

c = TrainingConfig(
    data_path="data/",  # denoised data from raddar
    labels_path="data/amex-default-prediction/train_labels.csv",  # original data
    train_feat_path="data/train_1_perc.parquet",
    test_feat_path="data/test_feat.parquet",  # aggregated features
    n_splits=5,
    log_path="xgb_log.md",
    submission_path="submission",
    n_trials=140,
    model=XGBClassifier,
    hyperparams=get_xgb_params,
)
t = AmexTrainer(config=c).setup().model_train().create_submission()
