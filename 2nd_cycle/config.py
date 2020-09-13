import numpy as np
import pandas as pd
from math import ceil
import logging
import logging.config
import json
import os
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_seq_items', None)

BASE_PATH = '~/HS'
EDA_PATH = os.path.join(BASE_PATH, 'data_analysis')
PREPROCESSED_PATH = os.path.join(BASE_PATH, 'preprocessing')
BACKUP_PATH = os.path.join(BASE_PATH, 'backup')
DATA_PATH = os.path.join(BASE_PATH, 'data')
FS_PATH = os.path.join(BASE_PATH, 'feature_engineering')
RESULT_PATH = os.path.join(BASE_PATH, 'result')

# name = [전체 데이터 컬럼]
names = []

# dtype = {미리 타입을 알고 정해줘야 함}
dtypes = {"PMT_TP": str}


## 예시
df_dtype = {, "CARD_TP" : np.object
, "EXP_YY" : np.string_
, "EXP_MM" : np.string_
, "BIN_NUM" : np.string_
, "PMT_TP" : np.string_}

def get_logger(cls_name):
    ## logger
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


class TrainOption():
    def __init__(self, opts):
        self.DATA_PATH = os.path.join(BASE_PATH, 'data')
        self.opts = opts
        self.set_path(self.opts)
        self.set_opts(self.opts)

    def set_path(self, op):
        self.BACKUP_PATH = os.path.join(BASE_PATH, 'backup', op)
        self.RESULT_PATH = os.path.join(BASE_PATH, 'result', op)

        self.TRAIN_DATA = os.path.join(self.BACKUP_PATH, op, 'train.csv')

        self.VAL_DATA = os.path.join(self.BACKUP_PATH, op, 'val.csv')

        self.TEST_DATA = os.path.join(self.BACKUP_PATH, op, 'test.csv')

        if op == 'RWN_model':
            self.TRAIN_DATA = os.path.join(BACKUP_PATH,'train.csv')
            self.VAL_DATA = os.path.join(BACKUP_PATH,'validaion.csv')
            self.RESULT_PATH = os.path.join(BASE_PATH, 'result')
            self.BACKUP_PATH = os.path.join(BASE_PATH, 'backup')
            self.FS_PATH = os.path.join(BASE_PATH, 'feature_selection')
        elif op == 'Semi_model':
            self.TRAIN_DATA = os.path.join(BACKUP_PATH,'train.csv')
            self.VAL_DATA = os.path.join(BACKUP_PATH,'validaion.csv')
            self.RESULT_PATH = os.path.join(BASE_PATH, 'result')
            self.BACKUP_PATH = os.path.join(BASE_PATH, 'backup')
            self.FS_PATH = os.path.join(BASE_PATH, 'feature_selection')
        elif op == 'DNN_model':
            self.TRAIN_DATA = os.path.join(BACKUP_PATH,'train.csv')
            self.VAL_DATA = os.path.join(BACKUP_PATH,'validaion.csv')
            self.RESULT_PATH = os.path.join(BASE_PATH, 'result')
            self.BACKUP_PATH = os.path.join(BASE_PATH, 'backup')
            self.FS_PATH = os.path.join(BASE_PATH, 'feature_selection')

        ## Normalizer
        self.NORMALIZER_PATH = os.path.join(self.RESULT_PATH, 'scaler_normalizer.pkl')

    def set_opts(self, op):
        if op == 'RWN_model':
            self.desc = '랜덤 와이어 네트워크'
        elif op == 'Semi_model':
            self.desc = 'Semi_GAN'
        elif op == 'DNN_model':
            self.desc = '일반딥러닝 '
        else:
            raise ValueError('invalid option')

BATCH_SIZE = 128
RANDOM_STATE = 42
EPOCHS = 50
VAL_RATIO = 0.1