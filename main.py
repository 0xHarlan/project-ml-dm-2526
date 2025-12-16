import logging
import time
import pandas as pd
import numpy as np
import optuna
import os
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from src.utils import *
from src.graphs import *
from src.features import *
from src.models import *
from src.metrics import *

# Logger
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Constatns
DATA_DIR = Path("datasets")
X_TRAIN_PATH = DATA_DIR / "x_train_T9QMMVq.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train_R0MqWmu.csv"
X_TEST_PATH  = DATA_DIR / "x_test_9F13O5s.csv"
SUBMISSION_PATH = "submission.csv"

# Hyper parameters optimization using optuna (Bayes-like approach). To try tuning the model at its maximum.
def optuna_hyperparameters_optimization(x_train, y_train, n_trials=300):
    """ 
    source & guidance used: 
    - https://medium.com/@siddhijadhav98/understanding-of-optuna-a-machine-learning-hyperparameter-a655c4301bcc
    - https://optuna.readthedocs.io/en/stable/reference/index.html
    """
    def objective(trial):
        # We initialize the ranges of main parameters, most are in ranges to avoid overfitting (except max_depth where values <15 seems to be risky)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 40),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
            'max_samples': trial.suggest_float('max_samples', 0.4, 0.8),    
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
            'n_jobs': 4,
            'random_state': 42
        }

        model = RandomForestRegressor(**params) 
        # We use a simple_validate (80/20 split) even if a cross-validate would be better because it's too time/energy consuming otherwise
        # Also, the ranges are accurate enough to avoid too much overfitting
        score = simple_validate_model(model, x_train, y_train)
        return score

    print(f"Starting optimizing with {n_trials} trials...")
    
    study = optuna.create_study(direction="minimize")

    # Reference value
    study.enqueue_trial({
            'n_estimators': 110,
            'max_depth': 35,
            'min_samples_split': 11,
            'min_samples_leaf': 14,
            'max_samples': 0.6,
            'max_features': 0.3
        })

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Finished !") 
    print(f"Best score: {study.best_value}") 
    print(f"Best params: {study.best_params}")

    with open(filename, 'w') as f:
        pprint(study.best_params, stream=f)
    print(f"Saved in {filename}")

    return study.best_params

def simple_validate_model(model, x_train, y_train):
    model_copy = clone(model) # we make sure we use a copy of the model

    x_train_noid = x_train.drop(columns = ["ID"])
    y_train_noid = y_train.drop(columns = ["ID"])

    #x_train_split, x_test_split, y_train_split, y_test_split = split(x_train_noid, y_train_noid, 0.2) # old split function
    x_train_split, x_test_split, y_train_split, y_test_split = humidity_split(x_train_noid, y_train_noid, 0.2)

    model_copy.fit(x_train_split, y_train_split)
    predictions = model_copy.predict(x_test_split)

    score = challenge_metric(predictions, y_test_split.values)

    return score

def cross_validate_model(model, x_train, y_train, folds = 5, engineer = False):
    LOGGER.info(f"Starting {folds}-fold (on humidity groups) cross-validation...")

    X = x_train.drop(columns = ["ID"])
    Y = y_train.drop(columns = ["ID"])

    if (engineer):
        X = feature_engineering(X)

    groups = pd.qcut(X['Humidity'], q=folds, labels=False)
    gkf = GroupKFold(n_splits=folds)

    scores = []
    i = 1
    for train_idx, val_idx in gkf.split(X, Y, groups=groups):

        X_train_fold = X.iloc[train_idx]
        Y_train_fold = Y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        Y_val_fold = Y.iloc[val_idx]

        model_copy = clone(model)
        humidity_mean = X_val_fold['Humidity'].mean()

        model_copy.fit(X_train_fold, Y_train_fold)
        predictions = np.clip(model_copy.predict(X_val_fold), 0, 1)

        score = challenge_metric(predictions, Y_val_fold.values)
        scores.append(score)

        LOGGER.info(f"  Fold {i}/{folds} - Humidity mean: {humidity_mean:.3f} -> Score: {score:.6f}")
        i+=1

    mean_score = np.mean(scores)
    std = np.std(scores)

    LOGGER.info(f"Cross-validation terminée: {mean_score:.6f} ± {std:.6f}")
        
    return {
        'scores': scores,
        'mean': mean_score,
        'std': std
    }

def generate_submission(model, x_train, y_train, x_test, engineer=False):
    LOGGER.info(f"Generating submission for model: {model}")

    X_train_full = x_train.drop(columns=["ID"])
    Y_train_full = y_train.drop(columns=["ID"])
    X_test_full  = x_test.drop(columns=["ID"])

    if engineer:
        LOGGER.info("Feature engineering...")
        X_train_full = feature_engineering(X_train_full)
        X_test_full  = feature_engineering(X_test_full)

    model = clone(model)

    model.fit(X_train_full, Y_train_full)

    predictions = model.predict(X_test_full)
    predictions = np.clip(predictions, 0, 1)

    result = pd.DataFrame(predictions, columns=y_train.columns[1:])
    result.insert(0, 'ID', x_test['ID'])

    result.to_csv("submission.csv", index = False)

# Sandbox function to tryout things, left "as is" for teacher review
def test(x_train, y_train):
    start = time.time()
    default_model = get_model("rfr_overfit")
    xgboost_model = get_model("xgboost_safe")

    print(simple_validate_model(default_model, x_train, y_train))
    print(cross_validate_model(default_model, x_train, y_train))
    print(cross_validate_model(default_model, x_train, y_train, engineer=True))

    # print(f"Default model + train/test split 80/20: {simple_validate_model(default_model, x_train, y_train)}")
    # print(f"XGBoost model + train/test split 80/20: {simple_validate_model(xgboost_model, x_train, y_train)}")

    # print(f"Default model + k-folding cross validation: {cross_validate_model(default_model, x_train, y_train, engineer=True)}")
    # print(f"XGBoost model + k-folding cross validation: {cross_validate_model(xgboost_model, x_train, y_train, engineer=True)}")
    
    # print("Cross validate ENGINEERING: ")
    # cross_validate_model(default_model, x_train, y_train, engineer=True)

    # print("Cross validate opti ENGINEERING: ")
    # cross_validate_model(default_model_opti, x_train, y_train, engineer=True)

    # NO / YES - Default model
    # 0.179330 / 0.179256 -> Mean/Std
    # 0.179330 / 0.181900 -> Mean/Std/S1-ratio
    
    # NO / YES - Default model tuned
    # 0.149028 / 0.155... -> Mean/Std/Humidity-ratio -> PUBLIC SCORE: 0,14587882872348712 !!

    # NO / YES - Xgboost
    # ... -> Not better than default_tuned

    # NO / YES - Default model tuned
    # 0.149... / 0.153321 -> mean/std/hum_ratio(b2)
    # 0.149..  / 0.154202 -> mean/std/hum_ratio(b1+b2)/s1_ratio(b1+b2)
    # ...      / 0.148702 -> humidity removal

    end = time.time() 
    print(f"Test() - Execution time: {end - start:.4f} seconds")

def main():
    x_train, y_train, x_test = read_datasets(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH)
    print(
        f"len(x_train): {len(x_train)}\n"
        f"len(y_train): {len(y_train)}\n"
        f"len(x_test): {len(x_test)}"
    )

    show_graphs(x_train, y_train, x_test)

    #test(x_train, y_train)
    #generate_submission(get_model("rfr_best_submission"), x_train, y_train, x_test, engineer=True)
    #optuna_hyperparameters_optimization(x_train, y_train)
    


if __name__ == "__main__":
    main()