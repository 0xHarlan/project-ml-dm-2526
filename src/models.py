from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Common parameters

SEED_THREADS = {
    "random_state": 42,
    "n_jobs": -1 # We use all threads by default
}

GPU_ACCEL = { # Params xgboost gpu acceleration (not very efficient on the data)
    "tree_method": "hist",
    "device": "cuda",
    "verbosity": 0
}

# Factory method to get the desired model
def get_model(model_name: str):
    match model_name:
        case "rfr_default": # Default benchmark model provided by the challenge
            return RandomForestRegressor(n_estimators=5, max_depth=7, min_samples_split=0.01, min_samples_leaf=30, **SEED_THREADS)

        case "rfr_tuned":   # Slightly tuned default model
            return RandomForestRegressor(n_estimators=100, max_depth=15, max_samples=0.5, min_samples_leaf=10, max_features=0.8, **SEED_THREADS)

        case "rfr_overfit":
            return RandomForestRegressor(n_estimators = 200, max_depth = None, min_samples_split = 2, min_samples_leaf = 1, random_state = 42, n_jobs = -1)    

        case "rfr_best_submission": # Model with best submission result!
            return RandomForestRegressor(n_estimators=110, max_depth=35, min_samples_split=11, min_samples_leaf=14, max_samples=0.6, max_features=0.3, **SEED_THREADS)

        case "rfr_opti": # Model with best results on y_train
            return RandomForestRegressor(n_estimators=179, max_depth=14, min_samples_split=6, min_samples_leaf=20, max_samples=0.6, max_features=0.3, **SEED_THREADS)

        # Attempts of using xgboost instead of RandomForestRegressor, without success

        case "xgboost_old":
            return XGBRegressor(n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, **SEED_THREADS, **GPU_ACCEL)

        case "xgboost":
            return XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, **SEED_THREADS, **GPU_ACCEL)

        case "xgboost_fast": 
            return XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, **SEED_THREADS, **GPU_ACCEL)

        case "xgboost_safe": # An attempt of reducing xgboost overfitting
            return XGBRegressor(n_estimators=250, max_depth=15, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0, **SEED_THREADS, **GPU_ACCEL)

        case _:
            raise ValueError(f"Unknown model: {model_name}")


