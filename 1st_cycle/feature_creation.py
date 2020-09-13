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
from threading import Thread, RLock
lock = RLock()
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import multiprocessing as mp
from multiprocessing import Pool, Pipe, Process, Lock, Manager, current_process

from config import *
from utils import *
# from columns_list import *
from feature import *
from feature_creation_columns_list import *

def create_data(readfile):
    with lock:
        start = datetime.datetime.now().replace(microsecond = 0)
        logger.info("==============Feature creation start==============")
        logger.info("[start] : {}".format(str(start)))
        logger.info("DATA_PATH : {}".format(DATA_PATH))
        
        ##########################################
        # Config
        ##########################################
        
        func = args.func
        types = args.types
        if types == 'code':
            col = CODE_COLS
        elif types == 'continuos':
            col = AMT_TYPE
        tt = args.period
        kc = UNIQ_COLS
        
        ##########################################
        # Data Load 
        ##########################################
        logger.info('===== Setting =====')
        logger.info('[Functions] : {}'.format(func))
        logger.info('[Target Col method] : {}'.format(types))        
        logger.info('[Preiod] : {}'.format(tt))
        
        load_file = os.path.join(DATA_PATH, readfile)
        with open(load_file, 'rb')as f:
            df = pickle.load(f)
        df = df.reset_index(drop = True)
        logger.info("Data Shape : {}".format(df.shape))
        logger.info("Null Value is : {}".format(df.isnull().sum().sum()))
        ##########################################
        # feature Creation
        ##########################################
        if types == 'continuos':
            df = quantile_feature(df, col, logger)

        df = auto_feature(df = df
                          , key_col = kc
                          , target_col = col
                          , period = tt
                          , func = func
                          , types = types
                          , unique = args.unique
                          , logger = logger
                         )
    
        logger.info("[Initial Shape is] : {}".format(df.shape))
        for col in df.columns:
            if df[col].dtypes == 'object':
                df.drop(col, axis = 1, inplace = True)
        logger.info("[Object Shape drop is] : {}".format(df.shape))

        df = mem_ext(df)
        
        logger.info("Final Columns is : {}".format(df.shape))
        
        save_file = 'create_dataset_' + str(readfile[-6 :])
        save_path = os.path.join(DATA_PATH, save_file)
        with open(save_path, 'wb')as f:
            pickle.dump(df, f)
            
        csv_file = os.path.join(DATA_PATH, 'made.csv')
        df.to_csv(csv_file, mode = 'a+', index = False, header = False)
        logger.info("Final NaN Value is : {}".format(df.isnull().sum().sum()))
        logger.info("Columns is : {}".format(df.shape))
        USE_COL = df.columns.tolist()
        logger.info("Columns is : {}".format(len(USE_COL)))
        with open(os.path.join(RESULT_PATH, 'use_col.txt'), 'wb')as f:
            pickle.dump(USE_COL, f)

if __name__ == '__main__':
    logger = get_logger(os.path.basename('Feature_Create'))
    args = args()
    files = []
    for i in range(2):
        file = 'preprocessed_dataset_{}.pkl'.format(str(i).zfill(2))
        files.append(file)

    n_worker = 12
    try:
        p = Pool(processes = n_worker)
        p.map(create_data, files)
    finally:
        p.close()
        p.join
    