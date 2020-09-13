import pandas as pd
pd.set_option("display.max_columns", None)
import numpy as np


import os, sys, json, pickle, time, datetime, argparse, tqdm

from config import *
from feature import *
from utils import *

from sklearn.impute import SimpleImputer
from tqdm import tqdm
from multiprocessing import Pool
################ No pool in here ################

def transform(file):
    
    start = time.time()
    logger.info("============== Feature Selection start ==============")
    logger.info("[start] : {}".format(str(start)))
    logger.info("DATA_PATH : {}".format(DATA_PATH))
    
    ##########################################
    # Config
    ##########################################
    
    
    ##########################################
    # feature Selection
    ##########################################
#     with open(os.path.join(RESULT_PATH, 'use_col.txt'), 'rb')as f:
#         use_col = pickle.load(f)
#     column_order = list(use_col)
    
    df = pd.read_csv(save_file, index_col = 0)
    tqdm.pandas()
    print(df.shape)
    drop_col = ['target',
               "MODE(data.ITEPD_GRP_ID)"
                , "MODE(data.RECHCT_EXCT_ID_1)"
                , "MODE(data.RCMS_SBJT_ID)"
                , "MODE(data.AGORG_BZEXC_ID)"
                , "MODE(data.RECHCT_USE_TRSC_PFMC_ID)"
                , "MODE(data.AGRT_ID)"
                , "MODE(data.EVDC_PPS_ATCH_DOC_ID_1)"
                , "MODE(data.STP_DE)"
                , "MODE(data.RCMS_BSNS_ID)"
                , "MODE(data.AGRT_ORGN_ID)"
                , "MIN(data.CPRT_REG_NO_X)"
                , "MODE(data.AGRT_ORGN_ID)"]

    x = df.drop(drop_col, axis = 1)
    
    y = df['target']
    
    x.fillna(0, inplace=True)
    
    logger.info("type transform")

    x = x.astype(int)
    x = mem_ext(x)
    return x, y
    
def selection_feature(x, y, logger):
    
    scaler = args.scaler
    
    logger.info("scaling")
    if scaler == 'robust':
        x_std = RobustScaler().fit_transform(x)
    elif scaler == 'minmax':
        x_std = MinMaxScaler().fit_transform(x)
    else:
        x_std = StandardScaler().fit_transform(x)
    logger.info("scaling done")

    
    logger.info("Data Shape is : {} / {}".format(x.shape, y.shape))
    cor_support, cor_feature = cor(x, x_std, y, logger)
    chi_support, chi_feature = selectbest(x, x_std, y, logger)
    emb_lr_support, emb_lr_feature = logistic(x, x_std, y, logger)
    rfe_support, rfe_feature = rfe(x, x_std, y, logger)
    rf_support, rf_feature = rfmodel(x, x_std, y, logger)
    lgb_support, lgb_feature = lgbmodel(x, x_std, y, logger)
    make_result(cor_support, chi_support, emb_lr_support
                , rfe_support, rf_support, lgb_support, x, logger)
    
    logger.info("[After Feature Selection shape] : {}".format(df.shape))
    h, m, s = count_time(start)
    logger.info("Total Feature Selection : [{}:{}:{}]".format(h, m, s))
    logger.info("===== Feature Selection Done =====")
    
if __name__ == '__main__':
    def args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--scaler', type = str, default = 'minmax', choices = ['robust', 'standard', 'minmax'], help = 'scaler before select feature')
        args = parser.parse_args()
        return args

    logger = get_logger(os.path.basename('Feature_transform'))
    save_file = os.path.join(DATA_PATH, 'created_df.csv')
    args = args()
    n_worker = 8
    x = []
    y = []
    
    try:
        p = Pool(processes = n_worker)
        for x, y in p.map(transform, save_file):
            x.append(x)
            y.append(y)
    finally:
        p.close()
        p.join
    logger.info("End of pool, shape :{}".format(x.shape))
    logger.info("End of pool, shape :{}".format(y.shape))
    selection_feature(x, y, logger)
    
