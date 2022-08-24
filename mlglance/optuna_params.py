from typing import Any, Dict
from pydantic import BaseModel
from typing import Optional


class FakeOptunaTrial(BaseModel):
    # TODO: implement optuna trial objects
    id: int = 0

    def suggest_int(self, *args, **kwargs):
        return "fake_flag"

    def suggest_float(self, *args, **kwargs):
        return "fake_flag"

    def suggest_categorical(self, *args, **kwargs):
        return "fake_flag"


def get_lgbm_params(trial: Any) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 500, 6000, step=10),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 100, 3000),
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0, 1),
        "max_bins": trial.suggest_int("max_bins", 128, 1024),
        "random_state": 42,
        "n_jobs": -1,
        "verbose_eval": 100,
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),

        # "n_estimators": 5040,
        # "learning_rate": 0.011951537998571194,
        # "num_leaves": 149,
        # "min_child_samples": 309,
        # "reg_lambda": 42,
        # "colsample_bytree": 0.48771929747370824,
        # "max_bins": 946,
        # "random_state": 42,
        # "n_jobs": -1,
    }



def get_xgb_params(trial: Any) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 500, 5200, step=10),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.1, log=True),
        "random_state": 42,
        "n_jobs": -1,
        # "tree_method": "gpu_hist",
        # "gpu_id": 0,
        "verbose_eval": 100,
    }

def get_metaclassifier_params(trial:Any)->Dict[str,Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 500, 5200, step=10),

        "random_state": 42,
        "n_jobs": -1,
        #"verbose_eval": 100,
    }

