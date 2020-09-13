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
from columns_list import *
from preprocess import *

def preprocess(df):
#     with lock:
    logger = get_logger(os.path.basename('Preprocessing_ft'))
    start = datetime.datetime.now().replace(microsecond = 0)
    logger.info("============== Preprocess start==============")
    logger.info("[Start] : {}".format(str(start)))
    logger.info("[DATA_PATH] : {}".format(DATA_PATH))

    ##########################################
    # Config
    ##########################################
    amt_method = args.amt_method
    normal_method = args.normal_method
    nlp_method = args.nlp_method
    vec_size = args.vec_size
    code = CODE_COLS
    nl = NATURAL_LANGUAGE
    amt = AMT_TYPE
    flt = FLOAT_TYPE
    ##########################################
    # Start
    ##########################################

    logger.info('===== Setting =====')
    logger.info('[Amount Imputation method] : {}'.format(amt_method))
    logger.info('[Not Amount Imputation method] : {}'.format(normal_method))
    logger.info('[Natural Preprocessing method] : {}'.format(nlp_method))

    # load Data
#     load_file = os.path.join(DATA_PATH, readfile)
#     with open(load_file, 'rb')as f:
#         df = pickle.load(f)
    df = df.reset_index(drop = True)
    #Noise check and target label, drop columns

    df = drop_noise_row(df,logger)
    df = target_labeling(df,logger)

    df.drop(drop_col, axis = 1, inplace = True)

#         df = sampling(df)

    logger.info("[df Shape] : {}".format(df.shape))

    # Fill missing value
    df = non_amt_replace(df, logger, method = normal_method)
    df = amt_replace(df, logger, method = amt_method)

    # Type per preprocess
    # non-amt
    df = unique_data_preprocessing(df, logger)

    ## Encoding Stage
    logger.info("===== One-Hot Encoding start =====")
    start = time.time()
    logger.info("[Before One-Hot Encoding columns shape] : {}".format(df.shape[1]))
    code = 'HIRK_BSNS_CL_CD'
    value_list = ['$$$', '$$$', 
                  '$$$', '$$$', '$$$']
    df = get_dummie(df, code, value_list)
    del code, value_list

    ch_col = ['$$$', '$$$']
    for col in ch_col:
        value_list = ['Y', 'N', 'O']
        df = get_dummie(df, col, value_list)
    del ch_col, value_list

    logger.info("[After One-Hot Encoding columns shape] : {}".format(df.shape[1]))
    h, m, s = count_time(start)
    logger.info("Total time for One-Hot Encoding : [{}:{}:{}]".format(h, m, s))
    logger.info("===== One-Hot Encoding done =====")

    # NLP
    df = df.reset_index(drop = True)
    logger.info("[Reset index, Shape] : {}".format(df.shape))
    df = nlp_processing(
        df = df
        , nl_cols = nl
        , method = nlp_method
        , vec_size = vec_size
        , logger = logger
    )
    # AMT columns processing
    df = amt_data_preprocessing(df, amt)
    df = add_sign(df, flt)
    df = mem_ext(df)
    save_file = '$$$.csv'
    save_path = os.path.join(DATA_PATH, save_file)
    colcol = df.columns.tolist()
    
    with open(os.path.join(DATA_PATH, 'use_col.txt'), 'wb')as f:
        pickle.dump(colcol, f)

    df.to_csv(save_path, mode = 'a+', index = False, header = False)

if __name__ == '__main__':
    
    args = args()
    
    iterator = pd.read_csv(os.path.join(DATA_PATH, '$$$.csv'), chunksize = 50000, skipinitialspace = True, dtype = dtypes)
    max_processors = 6 
    pool = Pool(processes = max_processors)
    
    f_list = []
    for df in iterator:
        f = pool.apply_async(preprocess, [df])
        f_list.append(f)
        if len(f_list) == max_processors:
            for f in f_list:
                f.get()
                del f_list[:]
