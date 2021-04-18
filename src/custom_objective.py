import xgboost as xgb
import numpy as np

def grad(y_true, y_pred):
    return 2*np.sin(y_pred - y_true)

def hess(y_true, y_pred):
    return 2*np.cos(y_pred - y_true)

def cos_loss(y_true, y_pred):
    return grad(y_true, y_pred), hess(y_true, y_pred)

def cos_eval(y_true, y_pred):
    return np.cos(y_pred - y_true).mean()