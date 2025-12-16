import unittest
import numpy as np
import pandas as pd
import xgboost as xgb
from main import sample_error, challenge_metric, class_distribution

# A class for unit testing that ended up not being used as much as it should have
class UnitTests(unittest.TestCase):

    def test_sample_error(self):
        pred = np.array([0.5, 0.2])
        true = np.array([0.4, 0.3])
         
        result = sample_error(pred, true)
        
        self.assertTrue(result > 0.0)

    def test_y_error(self):
    	y_pred = np.array([[0.0, 1.0, 0.0],[0.33, 0.33, 0.33],[0.5, 0.5, 1.0]])
    	y_true = y_pred

    	result = challenge_metric(y_pred, y_true)

    	self.assertTrue(result == 0.0)

    def test_class_means(self):
        data = {
            'ID': [1, 2, 3, 4, 5],
            'c1': [0.0, 0.0, 0.0, 0.0, 1.0],
            'c2': [0.5, 0.5, 0.5, 0.5, 0.5],
            'c3': [1.0, 1.0, 1.0, 1.0, 1.0],
            'c4': [0.0, 0.25, 0.5, 0.75, 1.0],
            'c5': [0.1, 0.2, 0.3, 0.4, 0.5],
        }

        expected = {
            'c1': 20.0,
            'c2': 100.0,
            'c3': 100.0,
            'c4': 60.0,
            'c5': 20.0,
        }

        result = class_distribution(pd.DataFrame(data).drop(columns=["ID"]), 0.50)
        result_dict = dict(zip(result['class'], result['percentage']))
        for key, value in expected.items():
            self.assertAlmostEqual(result_dict.get(key, 0), value, places=2)

    # Not really a Unittest but a simple way to verify if xgboost correctly registers gpu acceleration on your platform
    def test_gpu_acceleration(self):
        # Source: https://stackoverflow.com/questions/70507099/how-to-check-if-xgboost-uses-the-gpu
        xgb_model = xgb.XGBRegressor(
            tree_method="hist",
            device="cuda"
        )
        X = np.random.rand(50, 2)
        y = np.random.randint(2, size=50)

        xgb_model.fit(X, y)

        self.assertIsNotNone(xgb_model)


if __name__ == "__main__":
    unittest.main()