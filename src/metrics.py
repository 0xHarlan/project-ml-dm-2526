import numpy as np

# Takes an array of predictions and an array of truth values, 
# outputs the score of the predictions using the metric provided by the challenge (RMSE)
def challenge_metric(y_pred, y_test):
    weights = np.where(y_test < 0.5, 1.0, 1.2)

    squared_errors = weights * (y_pred - y_test) ** 2

    return np.sqrt(squared_errors.mean())
