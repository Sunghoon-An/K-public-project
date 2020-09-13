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
tqdm.pandas()
def create_feature_columns(file):
    
    start = time.time()
    logger.info("============== Feature Selection start ==============")
    logger.info("[start] : {}".format(str(start)))
    logger.info("DATA_PATH : {}".format(DATA_PATH))
    logger.info("DATA NAME : {}".format(file))
    
    ##########################################
    # Config
    ##########################################
    scaler = args.scaler
    
    ##########################################
    # feature Selection
    ##########################################
#     with open(os.path.join(RESULT_PATH, 'use_col.txt'), 'rb')as f:
#         use_col = pickle.load(f)
#     column_order = list(use_col)
    
    df = pd.read_csv(file)
    print("RCMS_BSNS_ID" in df.columns)
    non_use_col = ["TTL_DVLM_PRID_END_DE",
                "USE_DE",
                "TRSC_DE",
                "STP_DE",
                "DVLM_STR_DE",
                "DVLM_END_DE",
                "USE_REG_DT",
                "TRSC_PFMC_REG_DT",
                "RCMS_BSNS_ID",
                "RCMS_SBJT_ID",
                "AGRT_ID",
                "ITEPD_GRP_ID",
                "AGORG_BZEXC_ID",
                "AGRT_ORGN_ID",
                "RECHCT_USE_ITEPD_ID",
                "RECHCT_USE_TRSC_PFMC_ID",
                "EVDC_PPS_ATCH_DOC_ID_1",
                "RECHCT_EXCT_ID_1",
                "BSNSR_REG_NO",
                "CPRT_REG_NO_X",
                "SPLR_BSNSR_REG_NO",
                "KED_CD",
                "BSTP_CD",
                "BSNS_CL_CD"
                , "MODE(data.ITEPD_GRP_ID)"
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
                , "MODE(data.AGRT_ORGN_ID)"
                , "MODE(data.BSNSR_REG_NO)"
                , "TTL_DVLM_PRID_STR_DE"]
    
    df.drop(non_use_col, axis = 1, inplace = True)
    print(df.shape)
    df.loc[df.target != 0, 'choice'] = 1
    df.loc[df.target == 0, 'choice'] = 0
    df.drop('target', axis = 1, inplace = True)
    df = df.rename(columns = {'choice' : 'target'})
    
    logger.info("target is {}".format(df.target.value_counts()))
    
    df.fillna(0, inplace=True)
    
    for col in tqdm(df.columns):
        if df[col].dtypes == 'object':
            df[col] = df[col].astype(int)
        elif df[col].dtypes == 'float':
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(int)
            
    tr_col = [col for col in df.columns if 'target' in col]
    print(tr_col)

    x = df.drop(tr_col, axis = 1)
    
    y = df['target']
    
    logger.info("type transform")
    
    
    
    x = mem_ext(x)
#     x = np.sign(x)
#     for col in x.columns:
#         x["{}_point".format(str(col))] = np.sign(x[col])
#         x.loc[x["{}_point".format(str(col))] == 1, "figure"] = 1
#         x.loc[x["{}_point".format(str(col))] == -1, "figure"] = 0
#         x.loc[x["{}_point".format(str(col))] == 0, "figure"] = 0
#         x.drop("{}_point".format(str(col)), axis = 1, inplace = True)
#         x[col] = abs(x[col]).astype(np.uint16)
#         if (x[col] < 0).all():
#             raise ValueError("Some value of {} is invalid.".format(col))
#         else:
#             pass
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

    logger = get_logger(os.path.basename('Feature choice3'))
    save_file = os.path.join(DATA_PATH, 'make_feature3.csv')
    args = args()
    create_feature_columns(save_file)
