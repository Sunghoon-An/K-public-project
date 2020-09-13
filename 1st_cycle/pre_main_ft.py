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
    value_list = ['KEPCO', 'U00086', 'U00268', 'CLUSTER', 'S1895', 'BS2003', 'KEMCO', 'U00267',
'U00286', 'B10013', 'B10795', 'B10801', 'B10050', 'B00004', 'B10775', 'B10804',
'B10143', 'B10055', 'B60004', 'B10046', 'B10080', 'B00233', 'B10048',
'B10107', 'B10806', 'B10015', 'B10059', 'B10307', 'B10057', 'B10049', 'B10807',
'B10035', 'B10054', 'B10082', 'B00039', 'B10811', 'B10081', 'B10809', 'B10796',
'B10721', 'B90002', 'B10061', 'B10791', 'B10009', 'B10067', 'B10729', 'B20034',
'B00569', 'B10075', 'B00566', 'B10805', 'S3300', 'R9230', 'S3331', 'R0114',
'S4360', 'S3433', 'R0116', 'S1900', 'R9100', 'S1293', 'R9200', 'S3320', 'S6037',
'P0010', 'S3232', 'S3301', 'S3131', 'S3900', 'S4370', 'R0117', 'S6083', 'S3345',
'S3304', 'S3249', 'S3753', 'S3332', 'S4420', 'S4380', 'S3335', 'S2921', 'R9300',
'S6058', 'S3432', 'S3101', 'S3133', 'R9400', 'S4330', 'R0130', 'S9010', 'S6077',
'S6061', 'S3740', 'S6076', 'S6042', 'S3257', 'S4410', 'S6066', 'R0103', 'P0029',
'R0061', 'S6067', 'S3438', 'S6063', 'P0037', 'S3132', 'S3201', 'S1110', 'S6060',
'P0007', 'S3155', 'S4320', 'R0107', 'S4340', 'S6068', 'S6064', 'P0028', 'S9012',
'S6062', 'S6078', 'S3100', 'G03TOP', 'B00038', 'B50054', 'B00572', 'B10062',
'B10085', 'S1602', 'S0000', 'P0035', 'R0132', 'P0083', 'P0036', 'P0067', 'P0068',
'P0059', 'S6127', 'P0034', 'P0065', 'P0081', 'R0106', 'P0066', 'P0086', 'B00037',
'B10713', 'B10065', 'B10014', 'B50035', 'P0105', 'R0150', 'P0095', 'P0106',
'P0102', 'P0103', 'R0172', 'P0112', 'P0104', 'B10147', 'B50027', 'B00883',
'B00624', 'B10822', 'B10091', 'K0119', 'K0013', 'K0087', 'K0129', 'K0014',
'K0086', 'S4350', 'K0153', 'K0124', 'P0033', 'K0127']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'SBJT_SE_CD'
    value_list = ['H', 'F', 'G']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'BSNS_SPCH_SE_CD'
    value_list = [10, 20]
    get_dummie(df, code, value_list)
    del code, value_list

    code = 'SBJT_STEP_CD_1'
    value_list = [0, 1, 2, 3, 4, 6]
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'AGRT_SE_CD'
    value_list = [1, 2, 3]
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'AGRT_ORGN_ROLE_SE_CD'
    value_list = ['C', 'D', 'E']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'AGRT_ORGN_ENPR_SCL_CD'
    value_list = ['6.0', '2.0', '5.0', '3.0', '7.0', '1.0', '4.0', '6', '5', '2', '3', '7', '4', '1', 'C', 'A']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'CARD_STMT_WAY_SE_CD'
    value_list = [1, 2]
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'TAXT_TP_SE_CD'
    value_list = ['C02002', 'C02003', 'C02001']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'ENPR_SCL_CD'
    value_list = [1, 2, 3, 4, 5, 6, 7]
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'ORGN_CL_CD'
    value_list = ['SG5004', 'SG5005', 'SG5008', 'SG5003', 'SG5022', 'SG5007', 'SG5020', 'SG5002', 'SG5010', 'SG5006', 'SG5009', 'SG5011', 'SG5019', 'SG5021', 'SG5024', 'SG5014', 'SG5016', 'SG5015', 'SG5013']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'ORGN_STT_SE_CD'
    value_list = ['SAB001', 'SAB015', 'SAB002', 'SAB012', 'SAB013', 'SAB019']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'PTC_ITEPD_CD'
    value_list = ['B0201', 'B0202', 'B0206', 'B0203', 'B0301', 'B0204', 'B0205', 'B0401', 'B0101', 'B0102', 'B0207']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'DPTC_ITEPD_CD'
    value_list = ['B0201009', 'B0201015', 'B0201017', 'B0201018', 'B0201019', 'B0201020',
'B0202002', 'B0202004', 'B0202005', 'B0202047', 'B0202010', 'B0202048',
'B0202011', 'B0202014', 'B0202022', 'B0202019', 'B0202028', 'B0202029',
'B0202032', 'B0202023', 'B0202024', 'B0202020', 'B0202035', 'B0202052',
'B0202038', 'B0202054', 'B0202060', 'B0202057', 'B0206003', 'B0206001',
'B0206002', 'B0206004', 'B0206005', 'B0206006', 'B0206007', 'B0203001',
'B0301001', 'B0204002', 'B0204003', 'B0205001', 'B0201001', 'B0201002',
'B0201003', 'B0201021', 'B0201004', 'B0202062', 'B0202046', 'B0202033',
'B0202049', 'B0202050', 'B0202051', 'B0202053', 'B0202056', 'B0201007',
'B0202059', 'B0202055', 'B0201005', 'B0202036', 'B0202061', 'B0202058',
'B0202015', 'B0202030', 'B0201010', 'B0201006', 'B0401001', 'B0202001',
'B0202021', 'B0101001', 'B0202037', 'B0202013', 'B0202012', 'B0201016',
'B0102001', 'B0201008', 'B0301002', 'B0207001']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'USE_AMT_SE_CD'
    value_list = [1, 2, 3]
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'TRSC_EVDC_SE_CD'
    value_list = ['E', 'T', 'C', 'R']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'VAT_HDLG_SE_CD'
    value_list = ['B01001', 'B01002', 'B01003']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'SPLR_BSNS_REG_STT_CD'
    value_list = ['A04000', 'A04001', 'A04010', 'A04009']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'SPLR_TAXT_TP_SE_CD'
    value_list = ['A05001', 'A05003', 'A05004', 'A05002', 'C02002', 'C02003', 'C02001']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'EXCT_STEP_SE_CD'
    value_list = [1, 2]
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'EXCT_KIND_SE_CD'
    value_list = ['B03000', 'B03003', 'B03006', 'B03007', 'B03002', 'B03001', 'B03009', 'B03004', 'B03005', 'B03010', 'B03011']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'RNM_IDEN_SE_CD'
    value_list = ['B06001', 'B06005', 'B06006', 'B06008', 'B06004', 'B06009', 'B06007']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'MRC_STMT_MTHD_SE_CD'
    value_list = ['A', 'B', 'C']
    df = get_dummie(df, code, value_list)
    del code, value_list

    code = 'FRGN_USE_SE_CD'
    value_list = ['A', 'B']
    df = get_dummie(df, code, value_list)
    del code, value_list

    YN_col = ['NOPRFT_ORGN_YN', 'LCOST_ACCT_USE_YN', 'UNIC_AORGN_YN', 'FRGN_ORGN_YN', 'ORGN_CNLK_DTRS_YN', 'ORGN_CNLK_DTRS_YN_1', 'SBJT_AGRT_ORGN_TRSC_YN', 'EXCR_ORGN_APRB_YN', 'CNCL_RCV_YN', 'CLSBS_YN', 'ENPR_CRDT_BAD_YN', 'ASLCOST_ITEM_YN', 'ETC_EVDC_PSBLT_YN', '정밀점검대상여부', '기관통합관리책임자_보유_여부', 'RCMS_시스템_연계기관_여부']
    for col in YN_col:
        value_list = ['Y', 'N']
        df = get_dummie(df, col, value_list)
    del YN_col, value_list

    ch_col = ['CHAC_TRSF_ALLW_YN', 'MDTR_EVDC_ECTN_ALLW_YN']
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
    save_file = 'preprocessed_ft.csv'
    save_path = os.path.join(DATA_PATH, save_file)
    colcol = df.columns.tolist()
    
    with open(os.path.join(DATA_PATH, 'use_col.txt'), 'wb')as f:
        pickle.dump(colcol, f)

    df.to_csv(save_path, mode = 'a+', index = False, header = False)

if __name__ == '__main__':
    
    args = args()
    
    iterator = pd.read_csv(os.path.join(DATA_PATH, 'ft.csv'), chunksize = 50000, skipinitialspace = True, dtype = dtypes)
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