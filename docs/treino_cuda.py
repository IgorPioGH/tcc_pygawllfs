import time

import numpy as np
import xgboost as xgb

X = np.random.rand(20_000_000, 30)
y = np.random.randint(0, 2, 20_000_000)

dtrain = xgb.DMatrix(X, label=y)
params = {"tree_method": "hist", "device": "cuda", "objective": "binary:logistic"}

t0 = time.perf_counter()
bst = xgb.train(params, dtrain, num_boost_round=100)
print("tempo GPU:", time.perf_counter() - t0)
