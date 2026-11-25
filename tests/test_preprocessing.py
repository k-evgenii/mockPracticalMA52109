###
## cluster_maker - tests for preprocessing.py
## <your name> - University of Bath
## November 2026
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.preprocessing import (
    select_features,
    standardise_features
)

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Two float features on very different scales (~1000x), not proportional
        self.X = np.array([
            [2, 1],
            [4, 3],
            [6, 6],
            [8, 9],
        ], dtype=float)

        self.X[:, 0] = self.X[:, 0] / 100     # 0.02 … 0.08
        self.X[:, 1] = self.X[:, 1] * 500     # 500 … 4500


    # Detects: standardisation applied to the whole array at once, along the wrong
    # axis, or not at all.
    def test_standardise_features_scales_each_column_independently(self):
        X_scaled = standardise_features(self.X)

        self.assertEqual(X_scaled.shape, self.X.shape)
        # Every column (feature) must have mean 0 and standard deviation 1
        self.assertTrue(np.allclose(X_scaled.mean(axis=0), 0.0))
        self.assertTrue(np.allclose(X_scaled.std(axis=0), 1.0))


    # Detects: a categorical column whose labels happen to be numbers (e.g. region
    # codes 1, 2, 3) being accepted as a numeric feature.

    # The check must rely on the column's dtype, and the error must name the offending column
    def test_select_features_rejects_numeric_looking_categorical_column(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "region": pd.Categorical([1, 2, 3, 1]),
        })

        with self.assertRaises(TypeError) as cm:
            select_features(df, ["x", "region"])

        message = str(cm.exception)
        self.assertIn("region", message)   # the bad column is named
        self.assertNotIn("'x'", message)   # the valid numeric column is not blamed

    # Detects: standardisation done in place (e.g. StandardScaler(copy=False), or
    # "X -= mean; X /= std"), which silently overwrites the caller's raw data with
    # scaled values.
    def test_standardise_features_does_not_modify_input(self):
        X_before = self.X.copy()

        X_scaled = standardise_features(self.X)

        # The caller's array must hold exactly the original values
        np.testing.assert_array_equal(self.X, X_before)
        # The result must be a new array, not the input's memory reused
        self.assertFalse(np.shares_memory(X_scaled, self.X))


if __name__ == "__main__":
    unittest.main()
