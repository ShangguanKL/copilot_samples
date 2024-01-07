from xgboost import XGBRegressor
from numpy import ndarray
import matplotlib.pyplot as plt
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

__XGBOOST_DEFAULT_PARAM__ = {
    "n_estimators": 1000,
    "max_depth": 8,
    "learning_rate": 0.05,
    "device": "cpu",
    "early_stopping_rounds": 20
}
DEBUG = False

class RomModelXgboost():
    def __init__(self, n_estimators: int, max_depth: int, learning_rate: float, 
                 device: str, early_stopping_rounds: int):
        self.model = XGBRegressor(
            n_estimators = n_estimators,
            max_depth = max_depth,
            learning_rate = learning_rate,
            device = device,
            early_stopping_rounds = early_stopping_rounds
        )

    def train(self, x_train: ndarray, y_train: ndarray, x_valid: ndarray, y_valid:ndarray):
        self.model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)], eval_metric='rmse')
        # self.model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)], eval_metric='error') # for classification
        self.dim_f = x_train.shape[-1]

    def predict_with_best(self, x_test: ndarray):
        return self.model.predict(x_test, iteration_range=(0, self.model.best_iteration+1))
    
    def save_model(self, fn: str):
        self.model.save_model(fn)
    
    def load_model(self, fn: str):
        self.model.load_model(fn)
    
    def convert_to_onnx(self, fn: str):
        initial_type = [('float_input', FloatTensorType([1, self.dim_f]))]
        onnx_model = onnxmltools.convert_xgboost(self.model, initial_types=initial_type)
        onnxmltools.utils.save_model(onnx_model, fn)
    
    def drow_rmse(self, fn: str):
        plt.figure(figsize=(12, 6))
        plt.title("xgboost_rmse")
        plt.plot(self.model.evals_result_["validation_0"]["rmse"], label="train")
        plt.plot(self.model.evals_result_["validation_1"]["rmse"], label="validation")
        plt.xlabel("iteration")
        plt.ylabel("rmse")
        plt.legend()
        # plt.show()
        plt.savefig(fn)
        plt.close()
        



if DEBUG:
    import pandas as pd
    import numpy as np
    df = pd.read_csv('_data.csv')
    data = {}
    for col in df.columns:
        data[col] = df[col].values
    x = np.column_stack([data['s1'], data['s2'], data["s3"]])
    y = np.column_stack([data['s4'], data['s5']])
    x_train = x[:700] 
    y_train = y[:700]
    x_valid = x[700:]
    y_valid = y[700:]
    rmx = RomModelXgboost(**__XGBOOST_DEFAULT_PARAM__)
    rmx.train(x_train, y_train, x_valid, y_valid)
    