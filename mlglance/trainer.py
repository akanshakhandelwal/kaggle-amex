# This notebook: https://www.kaggle.com/code/kmmohsin/lgbm-with-optuna-for-tuning-on-c-gpu-lb-0-796/notebook
# Features data: https://www.kaggle.com/datasets/kmmohsin/amex-denoised-aggregated-features
# Features notebook: https://www.kaggle.com/code/ambrosm/amex-lightgbm-quickstart
import gc
from inspect import stack
import json
import sys
import warnings
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Any, List, Optional, Union, Dict
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np  # type:ignore
import optuna
import pandas as pd  # type:ignore
import wandb
from lightgbm import log_evaluation
from loguru import logger
from pydantic import BaseModel
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from .training_config import TrainingConfig
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from .utils import amex_metric, format_md, format_stderr, lgb_amex_metric
from sklearn.ensemble import StackingClassifier
from .optuna_params import get_lgbm_params, get_xgb_params, FakeOptunaTrial
from sklearn.metrics import fbeta_score, make_scorer
wandb.init(project="amex", entity="akanshakhandelwal", name="amex-runs" + str(time()))



class AmexTrainer(BaseModel):
    config: TrainingConfig
    train: Optional[pd.DataFrame]
    test: Optional[pd.DataFrame]
    target: Optional[np.ndarray]
    features: Optional[np.ndarray]
    model: Any
    y_pred_list: Optional[np.ndarray] = []
    current_time: int = time()
    wandb_params_table: Any
    params_suggested: Optional[Dict]
    params_notsuggested: Optional[Dict]
    stack_classifier:Any

    class Config:
        arbitrary_types_allowed = True

    def generate_sample(
        self,
        *,
        sampling_frac: float,
        stratify_col: str,
        filename: Union[str, Path],
    ):
        train_labels = pd.read_csv(Path(self.config.labels_path))
        train_labels = train_labels.set_index("customer_ID")
        self.train = self.train.merge(train_labels, left_index=True, right_index=True)
        sample = self.train.groupby(
            stratify_col,
            group_keys=False,
        ).apply(lambda x: x.sample(frac=sampling_frac))
        sample.to_parquet(
            Path(self.config.data_path) / filename,
            engine="pyarrow",
        )

    def setup(self):
        """Run pre-training tasks (set up data, logger)"""

        self.train = pd.read_parquet(self.config.train_feat_path)
        self.test = pd.read_parquet(self.config.test_feat_path)
        self.target = pd.read_csv(self.config.labels_path).target.values
        #self.target = self.train.target
        self.features = [
            f for f in self.train.columns if f != "customer_ID" and f != "target"
        ]
        logger.debug(
            f"Shapes: target:{self.target.shape}, train:{self.train.shape}, test:{self.test.shape}"
        )
        logger.add(
            sys.stderr,
            colorize=True,
            format=format_stderr,
            enqueue=True,
        )
        logger.add(
            self.config.log_path,
            format=format_md,
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
        logger.debug("Data setup complete")
        return self

    def create_submission(self):
        """Create submission for the (kaggle) Amex Default prediction competition"""
        subm = pd.DataFrame(
            {
                "customer_ID": self.test.index,
                "prediction": np.mean(self.y_pred_list, axis=0),
            }
        )
        Path(self.config.submission_path).mkdir(parents=True, exist_ok=True)

        subm.to_csv(
            Path(self.config.submission_path) / "submission.csv",
            index=False,
        )
        logger.debug("Submission file generated")

    def run_kfold(self, is_test: bool = False) -> List[float]:
        """Fit model to data and return score"""
        score_list, y_pred_list = [], []
        kf = StratifiedKFold(n_splits=self.config.n_splits)

        for fold, (idx_tr, idx_va) in enumerate(kf.split(self.train, self.target)):
            X_tr, X_va, y_tr, y_va = (
                None,
                None,
                None,
                None,
            )  # TODO: is this line necessry?
            X_tr = self.train.iloc[idx_tr][self.features]
            X_va = self.train.iloc[idx_va][self.features]
            y_tr = self.target[idx_tr]
            y_va = self.target[idx_va]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric=[lgb_amex_metric],
                    callbacks=[log_evaluation(1000)],
                )
            X_tr, y_tr = None, None
            y_va_pred = self.model.predict_proba(X_va,raw_score=True)
            score = amex_metric(y_va, y_va_pred)
            n_trees = self.model.best_iteration_
            if n_trees is None:
                n_trees = self.model.n_estimators
            logger.debug(f"Fold {fold}, {n_trees:5} trees, Score = {score:.5f}")
            score_list.append(score)

        logger.debug(f"OOF Score: {np.mean(score_list):.5f}")
        if is_test:
            self.y_pred_list.append(
                self.model.predict_proba(self.test[self.features],raw_score=True)
            )
        return score_list

    def objective(self, trial: Any):
        """Optuna objective"""
        self.model = self.config.model(**self.config.hyperparams(trial=trial))
        score_list = self.run_kfold()
        logger.debug(f"Trial params: {trial.params}")
        param_tracked = {
            k: v for k, v in trial.params.items() if k in self.params_suggested
        }
        self.wandb_params_table.add_data(
            trial.number,
            np.mean(score_list),
            *param_tracked.values(),
            *self.params_notsuggested.values(),
        )

        return np.mean(score_list)

    def model_train(self):
        f = FakeOptunaTrial()
        self.params_suggested = {
            k: v
            for k, v in self.config.hyperparams(trial=f).items()
            if v == "fake_flag"
        }
        self.params_notsuggested = {
            k: v
            for k, v in self.config.hyperparams(trial=f).items()
            if v != "fake_flag"
        }
        logger.debug(f"Params being tuned by Optuna: {self.params_suggested}")
        logger.debug(f"Params NOT being tuned by Optuna {self.params_notsuggested}")
        self.wandb_params_table = wandb.Table(
            columns=["trial", "score"]
            + list(self.params_suggested.keys())
            + list(self.params_notsuggested.keys())
        )
        study = optuna.create_study(direction="maximize")
        objective = lambda trial: self.objective(trial)
        study.optimize(objective, n_trials=self.config.n_trials)

        logger.debug(f"Number of finished trials: {len(study.trials)}")
        logger.debug(f"Best trial value: {study.best_trial.value}")
        logger.debug(
            f"Params: {json.dumps(study.best_trial.params, indent=2, sort_keys=True)}"
        )
        gc.collect()

        self.model = self.config.model(**study.best_trial.params)
        score_list = self.run_kfold(is_test=True)
        logger.debug(f"score_list: {score_list}")
        wandb.log(
            {
                "best_score": np.mean(score_list),
                "best_trial": study.best_trial.number,
                "total_time": str(timedelta(seconds=time() - self.current_time)),
                **study.best_trial.params,
            }
        )
        logger.debug("Model training completed")
        wandb.log({"Hyperparameters": self.wandb_params_table})
        self.stack_classifier = self.model
        return self
        
    def random_forest_model(self):
       
        from sklearn.model_selection import RandomizedSearchCV
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        rf = RandomForestClassifier()
 
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1,scoring=make_scorer(amex_metric, greater_is_better=False))

        rf_random.fit(self.train[self.features], self.target) # Train model

    # Make predictions
        y_train_pred = rf_random.predict(self.train[self.features])
        score = amex_metric(self.target,y_train_pred)
        print("Random Forest Score", score)
        y_test_pred = rf_random.predict(self.test[self.features])
        score_1 = amex_metric(self.target1,y_test_pred)
        print("RF Model",score_1)
        return rf_random

        
    def lightgbm_model(self):
        
        self.config.model = LGBMClassifier
        self.config.hyperparams = get_lgbm_params
        self.model_train()
       
        return self.stack_classifier

              
    def xgb_model(self):
        
        self.config.model = XGBClassifier
        self.config.hyperparams = get_xgb_params
        self.model_train()
       
        return self.stack_classifier

    def stacked_model(self):
        estimator_list = [
               ('rf', self.xgb_model()),
               ('lgbmmodel',self.lightgbm_model())
                ]

    # Build stack model
        stack_model = sklearn.ensemble.StackingClassifier(
            estimators=estimator_list, final_estimator=LogisticRegression(),
            )

    # Train stacked model
        stack_model.fit(self.train[self.features], self.target)
        y_train_pred = stack_model.predict(self.train[self.features])
        score = amex_metric(self.target,y_train_pred)
        print("Stacked Model",score)
        score_test= stack_model.predict(self.test[self.features])
        score_1 = amex_metric(self.target_test,score_test)
        print("Stacked Model",score_1)
        return self