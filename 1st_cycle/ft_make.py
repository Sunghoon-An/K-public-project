import pandas as pd
pd.set_option("display.max_columns", None)
import numpy as np

import os, sys, json, pickle, time, datetime, argparse, logging

from config import *
from feature import *
from utils import *
import featuretools as ft
import featuretools.variable_types as vtypes

warnings.filterwarnings("ignore")
csv_file = os.path.join(DATA_PATH, 'preprocessed_ft.csv')

with open(os.path.join(DATA_PATH, 'use_col.txt'), 'rb')as f:
    use_col = pickle.load(f)
column_order = list(use_col)
df = pd.read_csv(csv_file, usecols = column_order, names = column_order, index_col = 0)
logger = get_logger(os.path.basename('Auto feature'))
start = time.time()
logger.info("==============Feature creation start==============")
logger.info("[start] : {}".format(str(start)))
logger.info("DATA_PATH : {}".format(DATA_PATH))

df = df.reset_index(drop = True)

orgn_bool = ["CHAC_TRSF_ALLW_YN",
"NOPRFT_ORGN_YN",
"ETC_EVDC_PSBLT_YN",
"MCOST_YN",
"MDTR_EVDC_ECTN_ALLW_YN",
"FRGN_ORGN_YN",
"LCOST_ACCT_USE_YN",
"CLSBS_YN",
"ENPR_CRDT_BAD_YN",
"AGRT_ORGN_ENPR_SCL_CD",
"AGRT_ORGN_ROLE_SE_CD",
"AGRT_SE_CD",
"SBJT_STEP_CD_1",
"BSNS_CL_CD",
"HIRK_BSNS_CL_CD",
"CARD_STMT_WAY_SE_CD",
"ENPR_SCL_CD",
"ORGN_CL_CD",
"ORGN_STT_SE_CD",
"TAXT_TP_SE_CD",
"BSNS_SPCH_SE_CD",
"SBJT_SE_CD",
"BSTP_CD",
"KED_CD"]
o_bool = []
for c in orgn_bool:
    k = [col for col in df.columns if c in col]
    o_bool+=k

orgn_ord = ["ANNL",
"ATDT_CNT",
"DFN_IT",
"EXAA_IT"]
o_ord = []
for c in orgn_ord:
    k = [col for col in df.columns if c in col]
    o_ord+=k

orgn_cont = ["AGRT_BNDS_AMT",
"GOV_CTRB_AMT_1",
"LCGVN_ALOT_CASH_AMT_1",
"LCGVN_ALOT_SPOT_AMT_1",
"ORGN_TTL_BSNS_AMT",
"PRVT_ALOT_CASH_AMT_1",
"PRVT_ALOT_SPOT_AMT_1",
"UPAY_GOV_CTRB_AMT_1",
"DEBT_PT",
"EQCP_PT",
"INT_RWRD_MLTP_PT",
"OPRFT_RT"
"BSNS_NM",
"BSTP_NM_X",
"BUCDT_NM",
"ORGN_NM",
"SBJT_NM",
"BSTP_NM_Y",
"ENPR_FORM_NM",
"ENPR_SCL_NM",
"RPRSR_NM",
"STP_FORM_CD_VL"]
o_cont = []
for c in orgn_cont:
    k = [col for col in df.columns if c in col]
    o_cont += k

etc_bool = ["RCMS_시스템_연계기관_여부",
"기관통합관리책임자_보유_여부",
"정밀점검대상여부",
"ASLCOST_ITEM_YN"]
e_bool = []
for c in etc_bool:
    k = [col for col in df.columns if c in col]
    e_bool += k

etc_ord = ["이체권한_보유자_숫자",
"인건비내역수"]
e_ord = []
for c in etc_ord:
    k = [col for col in df.columns if c in col]
    e_ord += k

etc_cont = ["기타증빙예외",
"자계좌이체구분",
"증빙파일명",
"청구확인여부"]
e_cont = []
for c in etc_cont:
    k = [col for col in df.columns if c in col]
    e_cont += k

target_bool = ["EXCR_ORGN_APRB_YN",
"CNCL_RCV_YN",
"ORGN_CNLK_DTRS_YN",
"ORGN_CNLK_DTRS_YN_1",
"SBJT_AGRT_ORGN_TRSC_YN",
"EXCT_KIND_SE_CD",
"EXCT_STEP_SE_CD",
"RNM_IDEN_SE_CD",
"FRGN_USE_SE_CD",
"MRC_STMT_MTHD_SE_CD",
"CASH_SPOT_SE_CD",
"DPTC_ITEPD_CD",
"ETC_EVDC_ECTN_CD",
"PTC_ITEPD_CD",
"USE_AMT_SE_CD",
"SPLR_BSNS_REG_STT_CD",
"SPLR_TAXT_TP_SE_CD",
"TRSC_EVDC_SE_CD",
"VAT_HDLG_SE_CD"]
t_bool = []
for c in target_bool:
    k = [col for col in df.columns if c in col]
    t_bool += k

target_cont = ["SECH_AMT",
"SPLY_AMT_1",
"SUM_AMT_1",
"TXMT",
"FAT_AMT",
"FCO_AMT",
"FEC_AMT",
"FPU_SUM_AMT",
"FSA_AMT",
"FSU_AMT",
"BILL_AMT",
"CASH_AMT",
"CHQE_AMT",
"CRPH_AMPT_AMT",
"SPLY_AMT",
"SUM_AMT",
"VAT_AMT",
"RSTO_CONF_CROV_SPLY_AMT",
"RSTO_CONF_CROV_VAT_AMT",
"RSTO_CONF_SPLY_AMT",
"RSTO_CONF_VAT_AMT",
"USE_CROV_SPLY_AMT",
"USE_CROV_VAT_AMT",
"USE_SPLY_AMT",
"USE_VAT_AMT",
"VAT_HDLG_RSTO_AMT",
"USE_SPLY_AMT_1",
"USE_VAT_AMT_1",
"PAY_USAG_NM",
"ART_NM",
"SPLR_BCMP_NM",
"SPLR_BSTP_NM",
"SPLR_BUCDT_NM"]
t_cont = []
for c in target_cont:
    k = [col for col in df.columns if c in col]
    t_cont += k

orgn_col = o_bool + o_ord + o_cont 
etc_col = e_bool + e_ord + e_cont
target_col = t_bool + t_cont

logger.info("[column checked]")

df['USE_DE'] = df['USE_DE'].apply(lambda x : pd.to_datetime(x, infer_datetime_format = True))

for v in (o_bool + e_bool + t_bool):
    df[v] = df[v].astype('bool')
for v in (o_cont + e_cont + t_cont):
    df[v] = df[v].astype(float)
for v in (o_ord + e_ord):
    try:
        df[v] = df[v].astype(int)
    except Exception as e:
        print('Could not convert {v} because of missing values.')


        
from collections import Counter
intersection = Counter(orgn_col) & Counter(target_col)
multiset_a_without_common = Counter(orgn_col) - intersection
multiset_b_without_common = Counter(target_col) - intersection
orgn_col = list(multiset_a_without_common.elements())
target_col = list(multiset_b_without_common.elements())

import collections
print(len(orgn_col))
print(len(etc_col))
print(len(target_col))
orgn_col = list(dict.fromkeys(orgn_col))
etc_col = list(dict.fromkeys(etc_col))
target_col = list(dict.fromkeys(target_col))
print(len(orgn_col))
print(len(etc_col))
print(len(target_col))
logger.info("data Columns before entity is : {}".format(df.shape))
es = ft.EntitySet(id = 'data_id')
es.entity_from_dataframe(entity_id = 'data'
                         , dataframe = df
                         , make_index = True
                         , index = 'target_id'
                         , time_index = 'USE_DE')

es.normalize_entity(base_entity_id = 'data'
                   , new_entity_id = 'orgn'
                   , index = 'RCMS_BSNS_ID'
                   , make_time_index = False
                   , additional_variables = orgn_col)

es.normalize_entity(base_entity_id = 'data'
                   , new_entity_id = 'target'
                   , index = 'RECHCT_USE_ITEPD_ID'
                   , make_time_index = False
                   , additional_variables = target_col + ['target'])
from featuretools.variable_types import Numeric, PandasTypes
from featuretools.primitives import make_agg_primitive

def range_calc(numeric):
    return np.max(numeric) - np.min(numeric)
range_ = make_agg_primitive(function = range_calc
                           , input_types = [PandasTypes]
                           , return_type = PandasTypes)
def p_corr_calc(numeric1, numeric2):
    return np.corrcoef(numeric1, numeric2)[0, 1]

pcorr_ = make_agg_primitive(function = p_corr_calc,
                            input_types = [PandasTypes, PandasTypes]
                            , return_type = PandasTypes)
def s_corr_calc(numeric1, numeric2):
    return spearmanr(numeric1, numeric2)[0]

scorr_ = make_agg_primitive(function = s_corr_calc, 
                           input_types = [PandasTypes, PandasTypes]
                           , return_type = PandasTypes)

feature_matrix, feature_names = ft.dfs(entityset = es
                                      , target_entity = 'target'
                                      , agg_primitives = ['min', 'max', 'mean', 'percent_true', 'num_unique'
                                                          , 'all', 'any', 'mode'
                                                         , 'sum', 'skew', 'std'
                                                          , range_, pcorr_, scorr_
                                                         ]
                                      , trans_primitives = ['percentile', 'day', 'haversine', 'weekday', 'year', 'month']
                                      , n_jobs = -1
                                       , max_depth = 2
                                      , verbose = 1
                                      , max_features = 1000)

fc = feature_matrix.columns.tolist()
dc = df.columns.tolist()

from collections import Counter
intersection = Counter(fc) & Counter(dc)
multiset_a_without_common = Counter(fc) - intersection
fc = list(multiset_a_without_common.elements())
feature_matrix = feature_matrix[fc]
logger.info("feature matrix Columns after entity is : {}".format(feature_matrix.shape))
new_df = pd.merge(df, feature_matrix, on = 'RECHCT_USE_ITEPD_ID', how = 'left')
new_df.drop('target_id', axis = 1, inplace = True)

csv_file = os.path.join(DATA_PATH, 'make_feature.csv')
df.to_csv(csv_file)

h, m, s = count_time(start)
logger.info("Total AutoFeature Creation : [{}:{}:{}]".format(h, m, s))
logger.info("===== AutoFeature Creation Done =====")