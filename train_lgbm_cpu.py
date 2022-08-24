from mlglance.trainer import AmexTrainer
from mlglance.training_config import TrainingConfig
from mlglance.optuna_params import get_lgbm_params
from lightgbm import LGBMClassifier


c = TrainingConfig(
    data_path="data/",  # denoised data from raddar
    labels_path="data/amex-default-prediction/train_labels.csv",  # original data
    train_feat_path="data/train_feat.parquet",
    test_feat_path="data/test_feat.parquet",  # aggregated features
    n_splits=5,
    log_path="lgbm_log.md",
    submission_path="submission-lgbm",
    n_trials=1,
    model=LGBMClassifier,
    hyperparams=get_lgbm_params,
)
t = AmexTrainer(config=c).setup().model_train().create_submission()
