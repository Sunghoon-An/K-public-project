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

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', default = 0.2
                        , type = float
                        , help = ' train test random size')
    parser.add_argument('--train_path', type = str, default = None
                        , help = 'train save format')
    parser.add_argument('--test_path', type = str, default = None
                        , help = 'validation save format')
    parser.add_argument('--k_size', default = 10
                        , type = int
                        , help = 'K-Fold k size')
    args = parser.parse_args()
    return args

def create_data(readfile):
    with lock:
        start = datetime.datetime.now().replace(microsecond = 0)
        logger.info("============== Make train - test start ==============")
        logger.info("[start] : {}".format(str(start)))
        logger.info("DATA_PATH : {}".format(DATA_PATH))
        
        ##########################################
        # Config
        ##########################################
        
        ratio = args.ratio
        k_size = args.k_size
        
        ##########################################
        # Data Load 
        ##########################################
        
        load_file = os.path.join(DATA_PATH, readfile)
        with open(load_file, 'rb')as f:
            df = pickle.load(f)
        df = df.reset_index(drop = True)
        
        x = df.drop('target', axis = 1)
        y = df['target']
  
        skf = StratifiedKFold(n_splits = k_size)
        for i, train_index, test_index in enumerate(skf.split(x, y)):
            train, test = train[train_index], train[test_index]
            fold_dir = os.path.join(FOLD_BASE_PATH, "fold_{}".format(i))
            if os.path.isdir(fold_dir) is False:
                os.makedirs(fold_dir)
        
            if args.train_path is not None:
                train.to_csv(os.path.join(fold_dir, args.train_path)
                             , mode = 'a+', index = False, header = False)

            if args.test_path is not None:
                test.to_csv(os.path.join(fold_dir,args.test_path)
                            , mode = 'a+', index = False, header = False)

if __name__ == '__main__':
    logger = get_logger(os.path.basename('feature_make'))
    args = args()
    files = []
    for i in range(2):
        file = 'preprocessed_dataset_test_{}.pkl'.format(str(i).zfill(2))#'create_dataset_test_{}.pkl'.format(str(i).zfill(2))
        files.append(file)

    n_worker = 2
    try:
        p = Pool(processes = n_worker)
        p.map(create_data, files)
    finally:
        p.close()
        p.join