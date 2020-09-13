import re, datetime
import pandas as pd
from pandas.api.types import CategoricalDtype
import pickle
import math
import numpy as np
from columns_list import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, recall_score, precision_score
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import ADASYN, SVMSMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

import os
import logging
import category_encoders as ce
import time
from config import *
import tensorflow.keras.backend as K
from multiprocessing import Pool

import tensorflow as tf
from tensorflow import keras

def get_logger(cls_name):
    logger = logging.getLogger(cls_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(cls_name.split('.')[0] + '.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.INFO)
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    
    return logger

def split_kfold(data,logger,num_k):
    logger.info("========================= Stratified KFold start =========================".format(method))
    logger.info("=========== K is {} ===========".format(num_k))
    start = time.time()
    train_data, test_data = train_test_split(data,test_size=0.2,statify=data["target"])
    skf = StratifiedKFold(n_splits=num_k)
    for i,train_index,val_index in enumerate(skf.split(train_data["target"])):
        train_data, val_data = train_data[train_index],train_data[val_index]
        fold_dir=os.path.join(FOLD_BASE_DIR,"fold_{}".format(i))
        if os.path.isdir(fold_dir) is False:
               os.makedirs(fold_dir)
        train_data.to_csv(os.path.join(fold_dir,"train_data.csv".format(i)),index=False)
        val_data.to_csv(os.path.join(fold_dir,"val_data.csv".format(i)),index=False)
    for i,_,test_index in enumerate(skf.split(test_data["target"])):
        real_val_data=test_data[test_index]
        fold_dir=os.path.join(FOLD_BASE_DIR,"fold_{}".format())
        real_val_data.to_csv(os.path.join(fold_dir,"fold_{}_real_val_data.csv".format(i)),index=False)
    end=time.time()
    a,b=train_data.shape
    c,d=real_val_data.shape  
    logger.info("===== train data shape : ({}, {}), validation data shape : ({}, {}) =====".format(a,b,c,d))
    logger.info("========================= taken time for Stratified KFold : {}sec =========================".format(end-start))


def mem_ext(df):
    def mem_usage(pd_obj):
        if isinstance(pd_obj, pd.DataFrame):
            us_b = pd_obj.memory_usage(deep = True).sum()
        else:
            us_b = pd_obj.memory_usage(deep = True)
        us_mb = us_b / 1024 ** 2
        return "{:03.2f}MB".format(us_mb)
    df_int = df.select_dtypes(include = ['int'])
    cnv_int = df_int.apply(pd.to_numeric, downcast = 'unsigned')
    print("int memory is : {}".format(mem_usage(df_int)))
    print("convert memory is : {}".format(mem_usage(cnv_int)))
    
    df_flt = df.select_dtypes(include = ['float'])
    cnv_flt = df_flt.apply(pd.to_numeric, downcast = 'unsigned')
    print("int memory is : {}".format(mem_usage(df_flt)))
    print("convert memory is : {}".format(mem_usage(cnv_flt)))
    
    del df_int, df_flt, cnv_int, cnv_flt
    return df

# def auc(y_test, y_pred):
#     auc = tf.metrics.auc(y_test, y_pred
#                          , curve = 'PR'
#                          , summation_method = 'careful_interpolation'
#                          , num_thresholds = 0.5
#                         )[1]
#     K.get_session().run(tf.local_variables_initializer())
#     return auc

class IntervalEvaluation(keras.callbacks.Callback):
    def __init__(self, validation_data = (), interval = 1, savedir = None, file = None, logger = None):
        super(keras.callbacks.Callback, self).__init__()
        self.x_val, self.y_val = validation_data
        self.interval = interval
        self.file = file
        self.logger = logger
        if savedir is not None:
            self.savedir = savedir
        if file is not None:
            self.file = file
    def on_epoch_end(self, epoch, logs = {}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose = 0)
            recall = recall_score(self.y_val, y_pred >= 0.5)
            precision = precision_score(self.y_val, y_pred >= 0.5)
            
            self.logger.info(f'Evaluation - recall : {recall}, precision : {precision}')
            with open(os.path.join(self.savedir, self.file), 'a+')as f:
                if epoch == 0:
                    f.write("epoch, recall, precision"+"\n")
                f.write(f"{epoch + 1}, {recall}, {precision} "+"\n")
        else:
            pass

#####################recall precision

def recall(y_test, y_pred):
    tp = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    fp = K.sum(K.round(K.clip(y_test, 0, 1)))
    recall = tp / (fp + K.epsilon())
    return recall

def precision(y_test, y_pred):
    tp = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    fp = K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = tp / (fp + K.epsilon())
    return recall

def last_checkpoint(checkpoint_dir):
    checkpoints = [i for i in  os.listdir(checkpoint_dir) if "checkpoint" in i]
    checkpoints.sort(key = lambda s : os.path.getmtime(os.path.join(checkpoint_dir, s)))
    return os.path.join(checkpoint_dir, checkpoints[-1])
'''
############################## Use conditional gradient only ##############################
""" example
history = model.fit(
    x_train
    , y_train
    , batch_size = batch_size
    , validation_data = (x_test, y_test)
    , epochs = epochs
    , callbacks = [CG_get_weight_norm])
"""                        

def frobenius_norm(m):
    """this function is to calculate the frobenius norm of the matrix of all layer's weight.
    Args:
        m : is a list of weight params for each layers
    """
    total_reduce_sum = 0
    for i in range(len(m)):
        total_reduce_sum = total_reduce_sum + tf.math.reduce_sum(m[i]**2)
    norm = total_reduce_sum ** 0.5
    return norm

def CG_get_weight_norm:
    CG_frobenius_norm_of_weight = []
    CG_get_weight_norm = LambdaCallback(
        on_epoch_end = lambda batch
        , logs = CG_frobenius_norm_of_weight.append(frobenius_norm(model.trainable_weights).np())
        )
    return CG_get_weight_norm
'''
def focal_loss(gamma = 2., alpha = 4.):
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        '''
        Args:
            y_true {tensor} : ground truth label, shape of batch size, nb_class
            y_pred {tensor} : model's output, shape of batch size, nb_class

        keywordk Args:
            gamma {float} : default : 2.0
            alpha {float} : default : 4.0
        Returns:
            [tensor] : loss
        '''
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        f1 = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_f1 = tf.reduce_max(f1, axis = 1)
        return tf.reduce_mean(reduced_f1)
    return focal_loss_fixed
                        
def sampling(df):
        f = df[(df['target'] == 1.0) 
               | (df['target'] == 2.0)
               | (df['target'] == 3.0)
               | (df['target'] == 4.0)
               | (df['target'] == 5.0)
               | (df['target'] == 6.0)
               | (df['target'] == 7.0)
               | (df['target'] == 8.0)
               | (df['target'] == 9.0)
               | (df['target'] == 10.0)
               | (df['target'] == 11.0)
               | (df['target'] == 12.0)
               | (df['target'] == 13.0)
               | (df['target'] == 14.0)
               | (df['target'] == 15.0)
              ]
        n = df[df['target'] == 0.0].sample(frac = 0.3)
        df = pd.concat([f, n], axis = 0)
        del f, n
        return df

def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter = sidx)]
    

def imbalance(x_train, y_train, types, over, under, logger, jobs, reffer):
    
    cate_col = []
    for col in reffer.columns:
        if reffer[col].dtypes == 'int':
            cate_col.append(col)
    logger.info("[categorical columns] : {}".format(len(cate_col)))
    cate_c = column_index(reffer, cate_col)
            
    start = datetime.datetime.now().replace(microsecond = 0)
    logger.info("[type] : {}".format(types))
    logger.info("[start time] : {}".format(start))
    
    logger.info("[Oversampling method] : {}".format(over))
    if over == 'ros':
        overs = RandomOverSampler(random_state = RANDOM_STATE, sampling_strategy = 'auto', n_jobs = jobs)
    elif over == 'border_smote':
        overs = BorderlineSMOTE(random_state = RANDOM_STATE, sampling_strategy = 'auto', n_jobs = jobs)
    elif over == 'svmsmote':
        overs = SVMSMOTE(random_state = RANDOM_STATE, sampling_strategy = 'auto', n_jobs = jobs)
    elif over == 'cate_smote':
        overs = SMOTENC(random_state = 42, sampling_strategy = 'auto'
                     , categorical_features = cate_c, n_jobs = jobs)
    elif over == 'adasyn':
        overs = ADASYN(random_state = 42, sampling_strategy = 'auto', n_jobs = jobs)
    else:
        pass
#         raise ValueError('No support Oversampling.')
        
    logger.info("[Undersampling method] : {}".format(under))
    if under == 'rus':
        unders = RandomUnderSampler(random_state = 42, sampling_strategy = 'auto', n_jobs = jobs)
    elif under == 'tomek': #border_smote
        unders = TomekLinks(sampling_strategy = 'auto', n_jobs = jobs)
    elif under == 'centroid':
        unders = ClusterCentroid(random_state = 42, sampling_strategy = 'auto', n_jobs = jobs)
    elif under == 'one_sided':
        unders = OneSidedSelection(random_state = 42, sampling_strategy = 'auto', n_jobs = jobs)
    else:
        pass
        #         raise ValueError('No support Undersampling.')
    
    logger.info("[Before sampling shape] : {}".format(x_train.shape))
    if types == 'both':
        logger.info("============== {} / {} Pipeline start ==============".format(over, under))
        pp = make_pipeline(overs, unders)
        x_res, y_res = pp.fit_resample(x_train, y_train)
    elif types == 'over':
        x_res, y_res = overs.fit_resample(x_train, y_train)
    elif types == 'under':
        x_res, y_res = unders.fit_resample(x_train, y_train)
    elif types == 'None':
        x_res, y_res = x_train, y_train
    else:
        pass
#         raise ValueError('What is type???')
    logger.info("[After sampling shape] : {}".format(x_res.shape))
    end = datetime.datetime.now().replace(microsecond = 0)
    logger.info("[total time] : {}".format(end - start))
    
    return x_res, y_res

def scale_choose(option):
    if option == 'minmax':
        scaler = MinMaxScaler()
    elif option == 'robust':
        scaler = RobustScaler()
    elif option == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError('Scaler type does not exist.')
    return scaler

def random_split(x, y, ratio):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = ratio, random_state = 42, shuffle = True, stratify = y)
    x_train = pd.DataFrame(x_train, columns = x.columns.tolist())
    x_train = x_train.reset_index(drop = True)
    y_train = pd.DataFrame(y_train, columns = ['target'])
    y_train = y_train.reset_index(drop = True)
    x_test = pd.DataFrame(x_test, columns = x.columns.tolist())
    x_test = x_test.reset_index(drop = True)
    y_test = pd.DataFrame(y_test, columns = ['target'])
    y_test = y_test.reset_index(drop = True)
    train = pd.merge(x_train, y_train, how = 'outer', left_index = True, right_index = True)
    test = pd.merge(x_test, y_test, how = 'outer', left_index = True, right_index = True)
    print(train.shape)
    print(test.shape)
    return train, test
