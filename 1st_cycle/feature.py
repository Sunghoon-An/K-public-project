import pandas as pd
import numpy as np
import glob
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import random
from random import randint
import os, sys, json, pickle, time, datetime, logging
from collections import Counter
import argparse, re
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.utils import class_weight

from collections import Counter
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf

from config import *
from utils import *
# from columns_list import *
from feature_creation_columns_list import *

import warnings
tf.random.set_seed(RANDOM_STATE)
warnings.filterwarnings(action = 'ignore')

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type = str, nargs = '+', default = 'Sum', choices = ['Sum', 'Average', 'Min', 'Max'], help = 'rolling pandas function')
    parser.add_argument('--types', type = str, default = 'code', choices = ['code', 'continuos'], help = 'target code for rolling')
    parser.add_argument('--period', type = str, nargs = '+', choices = ['7d', '15d', '30d'], help = 'period of time for rolling')
    parser.add_argument('--unique', action = 'store_true', help = 'unique count if you want')
    args = parser.parse_args()
    return args


def quantile_feature(df, col, logger):
    ##### Need to amt columns, not others #####
    logger.info("===== Quantile feature Start =====")
    start = time.time()
    logger.info("[Before Quantile shape] : {}".format(df.shape))
    quantile = [25, 50, 75]
    drop_col = []
    for c in col:
        for i in quantile:
            cl = str(c) + "_" + str(i)
            df[cl] = df[c].quantile(i / 100)
            drop_col.append(cl)
            if i == 25:
                df.loc[(df[c] <= df[c + "_25"]), c + "_1st_yn"] = 1
                df[c + "_1st_yn"].fillna(0)
            elif i == 50:
                df.loc[(df[c] > df[c + "_25"]) & (df[c] <= df[c + "_50"]), c + "_2nd_yn"] = 1
                df[c + "_2nd_yn"].fillna(0)
            elif i == 75:
                df.loc[(df[c] > df[c + "_50"]) & (df[c] <= df[c + "_75"]), c + "_3rd_yn"] = 1
                df[c + "_3rd_yn"].fillna(0)
                df.loc[(df[c] < df[c + "_75"]), c + "_4th_yn"] = 1
                df[c + "_4th_yn"].fillna(0)
    
    logger.info("[After Quantile shape] : {}".format(df.shape))
    h, m, s = count_time(start)
    logger.info("Total Quantiling : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Quantile feature Done =====")
    
    return df
    
def auto_feature(df, key_col, target_col, period, func, types, unique = True, logger = None):
    '''
    Args:
        df : data frame
        key_col : unique or customer related
        target_col : something to able to vectoriz column
        period : period of time to rolling function
        func : able to continuos
        type : continuos or code
        unique : Not discrete(code only)
    '''
        ####### Code Unique count #######
    logger.info("===== Automated feature Creation Start =====")
    start = time.time()
    logger.info("[Before Automated feature Creation shape] : {}".format(df.shape))
    
    df['USE_DE'] = df['USE_DE'].apply(lambda x : pd.to_datetime(x, infer_datetime_format = True))
    df['USE_DE'] = df['USE_DE'].apply(lambda x : pd.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))

    for key in key_col:
        for col in target_col:
            if key != col:
                for tp in period:
                    if unique:
                        logger.info("[Method] : Uniuqe")
                        aggr_func = lambda arr : pd.Series(arr).nunique()
                        colname = str(key) + '_n_' + str(col) + '_cnt_' + str(tp)
                        df = df.assign(temp_col = pd.factorize(df[col])[0])
                        group_df = df.groupby(key, as_index = 'SQL-stype')['USE_DE', temp_col].rolling(window = tp, on = 'USE_DE').agg(agg.func)
                        group_df.index = group_df.index.droplevel(0)
                        group_df.drop('USE_DE', axis = 1, inplace = True)
                        df = df.merge(pd.DataFrame(group_df).rename(columns = {col : colname}), how = 'left', left_index = True, right_index = True)
                        df.drop('temp_col', axis = 1, inplace = True)
                        df[colname].replace({np.NaN : 0}, inplace = True)
                    else:
                        logger.info("[Method] : Un Unique, Type is {}".format(types))
                        aggr_func = lambda arr : pd.Series(arr).nunique()
                        colname = str(key) + '_n_' + str(col) + '_cnt_' + str(tp)
                        if types == 'code':
                            group_df = df.groupby(key, as_index = 'SQL-stype')['USE_DE', col].rolling(window = tp, on = 'USE_DE').agg(aggr_func)
                        elif types == 'continuos':
                            for f in func:
                                if f == 'Sum':
                                    logger.info("[Function] : Sum is running, Key : {} / Target : {} / Period : {}".format(key, col, tp))
                                    group_df = df.groupby(key, as_index = 'SQL-stype')['USE_DE', col].rolling(window = tp, on = 'USE_DE').sum()
                                elif f == 'Average':
                                    logger.info("[Function] : Average is running, Key : {} / Target : {} / Period : {}".format(key, col, tp))
                                    group_df = df.groupby(key, as_index = 'SQL-stype')['USE_DE', col].rolling(window = tp, on = 'USE_DE').mean()
                                elif f == 'Min':
                                    logger.info("[Function] : Min is running, Key : {} / Target : {} / Period : {}".format(key, col, tp))
                                    group_df = df.groupby(key, as_index = 'SQL-stype')['USE_DE', col].rolling(window = tp, on = 'USE_DE').min()
                                elif f == 'Max':
                                    logger.info("[Function] : Max is running, Key : {} / Target : {} / Period : {}".format(key, col, tp))
                                    group_df = df.groupby(key, as_index = 'SQL-stype')['USE_DE', col].rolling(window = tp, on = 'USE_DE').max()
                                else:
                                    raise ValueError("error feture creation")
                        else:
                            raise ValueError("Some argument is invalid.")
                        group_df.index = group_df.index.droplevel(0)
                        group_df.drop('USE_DE', axis = 1, inplace = True)
                        df = df.merge(pd.DataFrame(group_df).rename(columns = {col : colname}), how = 'left', left_index = True, right_index = True)
                        df[colname].replace({np.NaN : 0}, inplace = True)
                        logger.info("[Column Created] : {}".format(colname))
                        
    del group_df
    logger.info("[Before Impute NaN value] : {}".format(df.isnull().sum().sum()))

    df = df.fillna(np.nan)
    
    x_null = df.columns[df.isnull().any()].tolist()
    
    train_df = df[x_null].dropna(axis = 1, how = "any")
    target_columns = set(x_null) - set(train_df.columns)
    for col in target_columns:
        total_df = pd.concat([train_df,df[col]], axis=1)
        imputer = SimpleImputer(missing_values = np.nan, strategy="most_frequent")
        df[col] = imputer.fit_transform(df)
        
    logger.info("fitting imputer : {}".format(df.shape))
    
    logger.info("[After Automated feature Creation shape] : {}".format(df.shape))
    h, m, s = count_time(start)
    logger.info("Total Automated Creation : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Automated feature Creation Done =====")
    
    return df


def cor(X, x_std, y, logger):
    logger.info("===== Correlation feature Select =====")
    feature_name = X.columns.tolist()
    start = time.time()
    cor_list = []
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    cor_list = [0 if np.isnan(1) else i for i in cor_list]
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-500:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in feature_name]
    h, m, s = count_time(start)
    logger.info("Total Correlation feature Select : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Correlation feature Select Done =====")
    return cor_support, cor_feature

def selectbest(X, x_std, y, logger):
    ## choice if you want scaler
    logger.info("===== Chi2 feature Select =====")
    start = time.time()
    
    
    chi_choices = SelectKBest(f_classif, k = 500)
    chi_choices.fit(x_std, y)
    chi_support = chi_choices.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()
    
    h, m, s = count_time(start)
    logger.info("Total Chi2 feature Select : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Chi2 feature Select Done =====")
    
    return chi_support, chi_feature

def logistic(X, x_std, y, logger):
    logger.info("===== Logistic feature Select =====")
    start = time.time()

    emb_lr_select = SelectFromModel(LogisticRegression(penalty = 'l2', class_weight = 'balanced'
                                                      , n_jobs = -1
                                                      , verbose = 1), '1.25 * median')
    emb_lr_select.fit(x_std, y)
    emb_lr_support = emb_lr_select.get_support()
    emb_lr_feature = X.loc[:, emb_lr_support].columns.tolist()
    
    h, m, s = count_time(start)
    logger.info("Total Logistic feature Select : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Logistic feature Select Done =====")
    
    return emb_lr_support, emb_lr_feature

def rfe(X, x_std, y, logger):
    logger.info("===== Recursive feature Select =====")
    start = time.time()

    rfe_select = RFE(estimator = LogisticRegression(class_weight = 'balanced'
                                                    , penalty = 'l2'
                                                   , n_jobs = -1
                                                   , verbose = 1), n_features_to_select = 500, step = 10, verbose = 1)
    rfe_select.fit(x_std, y)
    rfe_support = rfe_select.get_support()
    rfe_feature = X.loc[:, rfe_support].columns.tolist()
    
    h, m, s = count_time(start)
    logger.info("Total Recursive feature Select : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Recursive feature Select Done =====")
    
    return rfe_support, rfe_feature

def rfmodel(X, x_std, y, logger):
    logger.info("===== Random Forest feature Select =====")
    start = time.time()
    

    rf_select = SelectFromModel(RandomForestClassifier(n_estimators = 500
                                                      , class_weight = 'balanced'
#                                                       , learning_rate = 0.0001
                                                      , n_jobs = -1
                                                      , verbose = 1), threshold = '1.25 * median')
    rf_select.fit(x_std, y)
    rf_support = rf_select.get_support()
    rf_feature = X.loc[:, rf_support].columns.tolist()
    
    h, m, s = count_time(start)
    logger.info("Total Random Forest feature Select : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Random Forest feature Select Done =====")
    
    return rf_support, rf_feature

def lgbmodel(X, x_std, y, logger):
    logger.info("===== LGBM feature Select =====")
    start = time.time()

    lgb = LGBMClassifier(n_estimators = 500
                         , learning_rate = 0.0001
                         , class_weight = 'balanced'
                         , n_jobs = -1
                         , verbose = 1
                        )
    lgb_select = SelectFromModel(lgb, threshold = '1.25 * median')
    lgb_select.fit(x_std, y)
    lgb_support = lgb_select.get_support()
    lgb_feature = X.loc[:, lgb_support].columns.tolist()
    
    h, m, s = count_time(start)
    logger.info("Total LGBM feature Select : [{}:{}:{}]".format(h, m, s))
    logger.info("===== LGBM feature Select Done =====")
    
    return lgb_support, lgb_feature

def make_result(cor_support, chi_support, emb_lr_support, rfe_support, rf_support, lgb_support, X, logger):
    feature_name = X.columns.tolist()
    feature_df = pd.DataFrame({"Feature" : feature_name
                               , "Pearson" : cor_support
                               , "Chi-2" : chi_support
                               , "Logistic" : emb_lr_support
                               , "Random Forest" : rf_support
                               , "Recursive" : rfe_support
                               , " LightGBM" : lgb_support
                              })
    feature_df["Total"] = np.sum(feature_df, axis = 1)
    feature_df = feature_df.sort_values(["Total", "Feature"], ascending = False)
    feature_df.index = range(1, len(feature_df) + 1)
    feature_name = feature_df[(feature_df["Total"] >= 5)]
    
    fcols = feature_name.Feature.tolist()
    logger.info("chosen column shape is : {}".format(len(fcols)))

    with open(os.path.join(RESULT_PATH, 'col3.txt'), 'wb')as f:
        pickle.dump(fcols, f)



