import logging
import tensorflow as tf
import keras.backend as K
from config import *
import pandas as pd
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
from sklearn.utils import shuffle

from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def mem_ext(df):
    def mem_usage(pandas_obj):
        if isinstance(pandas_obj, pd.DataFrame):
            usage_b = pandas_obj.memory_usage(deep = True).sum()
        else:
            usage_b = pandas_obj.memory_usage(deep = True)
        usage_mb = usage_b / 1024 ** 2
        return "{:03.2f}MB".format(usage_mb)
    df_int = df.select_dtypes(include = ['int'])
    cnv_int = df_int.apply(pd.to_numeric, downcast = 'unsigned')
#     print(mem_usage(df_int))
#     print(mem_usage(cnv_int))
    df_flt = df.select_dtypes(include = ['float'])
    cnv_flt = df_flt.apply(pd.to_numeric, downcast = 'unsigned')
#     print(mem_usage(df_flt))
#     print(mem_usage(cnv_flt))
    df[cnv_int.columns] = cnv_int
    df[cnv_flt.columns] = cnv_flt
    del df_int, df_flt, cnv_flt, cnv_int
    return df

def log_transform(data, columns):
    df_log = data[columns].apply(lambda x :np.log10(x + 1))
    df_log.columns = 'log_' + df_log.columns
    data = pd.concat([data, df_log], axis = 1)
    return data

def de_log_transform(data, columns):
    df_log = data[columns].apply(lambda x :round(10**(x)-1, 2))
    df_log.columns = 'log2_' + df_log.columns
    data = pd.concat([data, df_log], axis = 1)
    return data

def print_confusion_matrix(y_test, y_pred):
    tp, fn, fp, tn = confusion_matrix(y_test, y_pred, labels = [1,0]).ravel()
    print('=========================')
    print(format('* cut-off : ', '15s'), format(cut_off*1000, '<15.1f'))
    print("""
    {}{}{}
    {}{}{}
    {}{}{}""".format(format('* Predict\\True     ', '15s'), format(1, '10.0f'), format(0, '10.0f'),
                     format('1', '>15s'), format(tp, '10.0f'), format(fp, '10.0f'),
                     format('0', '>15s'), format(fn, '10.0f'), format(tn, '10.0f')
                    )
         )
    print(format('* recall : ', '15s'), format(tp / (tp + fn), '<15.4f'))
    print(format('* precision : ', '15s'), format(tp / (tp + fp), '<15.4f'))
    print(format('* f1-score : ', '15s'), format(2*tp / (2*tp + fp + fn), '<15.4f'))
    return None

