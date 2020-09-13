import os
import time

"""보완상의 문제로 data, root는 모두 $$$로 Masking"""

BASE_PATH = "/$$$"
CODE_PATH = os.path.join(BASE_PATH,"process")
DATA_PATH = os.path.join(BASE_PATH, 'data')
PREPROCESS_PATH = os.path.join(CODE_PATH,"preprocess")
NLP_PATH = os.path.join(CODE_PATH,"nlp")
SCALER_PATH = os.path.join(CODE_PATH,"scaler")
FOLD_BASE_PATH = os.path.join(CODE_PATH,"fold")
RESULT_PATH = os.path.join(CODE_PATH,"result")

RANDOM_STATE = 42

def count_time(start):
    sec = time.time() - start
    h = int(sec // (60 * 60))
    m = int((sec - (h * 60 * 60)) // (60))
    s = int((sec - (h * 60 * 60)) - (m * 60))
    return h, m, s
