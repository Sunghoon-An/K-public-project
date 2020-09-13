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

import multiprocessing as mp
from multiprocessing import Pool, Pipe, Process, Lock, Manager, current_process

from config import *
from utils import *

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tqdm import tqdm
import warnings
tqdm.pandas()
warnings.filterwarnings("ignore")

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', default = 0.2
                        , type = float, required = False
                        , help = ' train test random size')
    parser.add_argument('--train_path', type = str, default = None
                        , help = 'train save format')
    parser.add_argument('--test_path', type = str, default = None
                        , help = 'validation save format')
    parser.add_argument('--random', action = 'store_true'
                        , help = 'if restore, use it', required = False)
    parser.add_argument('--fold', default = 10, type = int
                        , help = 'number of K')
    args = parser.parse_args()
    return args

def create_data(readfile):
    start = datetime.datetime.now().replace(microsecond = 0)
    logger.info("============== Make train - test start ==============")
    logger.info("[start] : {}".format(str(start)))
    logger.info("DATA_PATH : {}".format(DATA_PATH))

    ##########################################
    # Config
    ##########################################

    ratio = args.ratio
    fold = args.fold

    ##########################################
    # Data Load 
    ##########################################

    df = pd.read_csv(save_file, index_col = 0, low_memory = False)

    with open(os.path.join(RESULT_PATH, 'col2.txt'), 'rb')as f:
        use_col = pickle.load(f)
    column_order = list(use_col)
    column_order.append('target')
    df = df[column_order]
    logger.info("[feature selected {} shape]".format(df.shape))

    df = df.reset_index(drop = True)
    logger.info(df.target.value_counts())

    df.fillna(0, inplace=True)
    
#     df.loc[df.target != 0, 'choice'] = 1
#     df.loc[df.target == 0, 'choice'] = 0
#     df.drop('target', axis = 1, inplace = True)
#     df = df.rename(columns = {'choice' : 'target'})

    for col in tqdm(df.columns):
        if df[col].dtypes == 'object':
            df[col] = df[col].astype(int)
        elif df[col].dtypes == 'float':
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(int)
    
    x = df.drop('target', axis = 1)
    
    y = df['target']
    logger.info(df.target.value_counts())
    

    logger.info("type transform")

#         x = x.progress_apply(lambda x : x.astype(int))
#         y = y.progress_apply(lambda y : y.astype(int))
    x = mem_ext(x)
    if args.random:
        train, test = random_split(x, y, ratio)
        if args.train_path is not None:
            logger.info("train target is :{}".format(train["target"].value_counts()))
            train.to_csv(os.path.join(FOLD_BASE_PATH, args.train_path))

        if args.test_path is not None:
            logger.info("test target is :{}".format(test["target"].value_counts()))
            test.to_csv(os.path.join(FOLD_BASE_PATH, args.test_path))

    else:
        mskf = StratifiedKFold(n_splits = fold, random_state = 42)
        for i, (train_index, test_index) in enumerate(mskf.split(x, y)):
            fold_dir = os.path.join(FOLD_BASE_PATH, "fold_{}".format(i))
            if os.path.isdir(fold_dir) == False:
                os.mkdir(fold_dir)


            train, test = df.iloc[train_index], df.iloc[test_index]


            if os.path.isdir(fold_dir) is False:
                os.makedir(fold_dir)

            if args.train_path is not None:
                train.to_csv(os.path.join(fold_dir, args.train_path))

            if args.test_path is not None:
                test.to_csv(os.path.join(fold_dir, args.test_path))
        
        
        

if __name__ == '__main__':
    logger = get_logger(os.path.basename('train_test_split_find2'))
    args = args()
    
    save_file = os.path.join(DATA_PATH, 'make_feature2.csv')
    create_data(save_file)