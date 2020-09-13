import scipy
import itertools
from itertools import tee, combinations
from scipy.stats import shapiro, kstest, anderson, normaltest, pearsonr, spearmanr, kendalltau, chi2_contingency, levene, ttest_ind, mannwhitneyu, f_oneway, kruskal, friedmanchisquare
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import os
import logging
import pandas as pd
import numpy as np

def get_logger(cls_name):
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

def normality_test(df, amt_col):
    logger = get_logger(os.path.basename(__file__))
    
    logger.info("[[[[[H0: 정규 분포가 아닙니다.]]]]]")
    logger.info("[[[[[H1: 정규 분포입니다.]]]]]")
    logger.info("======================== Normality test Start ========================")
    logger.info("data shape :{}".format(df.shape))
    
    for i in df[amt_col].columns:
        fr = df[df.F == 1]
        nm = df[df.F == 0]
        fr = fr[i]
        nm = nm[i]
        logger.info("\n\n===========================column : {}==============================\n\n".format(i))
        logger.info("======================== Shapiro & KS test Start ========================")
        logger.info("fr[{}] shape :{}".format(i, fr.shape))
        logger.info("nm[{}] shape :{}".format(i, nm.shape))
        if len(fr) < 2000:
            stat1, p1 = shapiro(fr)
        else:
            stat1, p1 = kstest(fr, 'norm')
        if len(nm) < 2000:
            stat2, p2 = shapiro(nm)
        else:
            stat2, p2 = kstest(nm, 'norm')
        
        if p1 < 0.05:
            logger.info("Shapiro & KS Result: Fraud[{}]는 정규 분포가 아닙니다 -- [p-value]: {}".format(i,round(p1, 4)))
            logger.info("********* Fraud[{}]는 정규 분포를 만들어야 합니다(log치환 등)! *********".format(i))
        else:
            logger.info("Fraud[{}]는 정규 분포입니다! [p-value]: {}".format(i, round(p1, 4)))
            logger.info("@@@@@@@@@ Fraud[{}]를 사용하세요. @@@@@@@@@".format(i))
        if p2 < 0.05:
            logger.info("Shapiro & KS Result: Normal[{}]는 정규 분포가 아닙니다 -- [p-value]: {}".format(i, round(p2, 4)))
            logger.info("********* Normal[{}]는 정규 분포를 만들어야 합니다(log치환 등)! *********".format(i))
        else:
            logger.info("Normal[{}]는 정규 분포입니다! [p-value]: {}".format(i, round(p2, 4)))
            logger.info("@@@@@@@@@ Normal[{}]를 사용하세요. @@@@@@@@@".format(i))
        del stat1, stat2, p1, p2
        
        ## Agostino Test
        logger.info("======================== Agostino test Start ========================")
        stat1, p1 = normaltest(fr)
        stat2, p2 = normaltest(nm)
        
        if p1 < 0.05:
            logger.info("Normaltest Result: Fraud[{}]는 정규 분포가 아닙니다 -- [p-value]: {}".format(i,round(p1, 4)))
            logger.info("********* Fraud[{}]는 정규 분포를 만들어야 합니다(log치환 등)! *********".format(i))
        else:
            logger.info("Fraud[{}]는 정규 분포입니다! [p-value]: {}".format(i, round(p1, 4)))
            logger.info("@@@@@@@@@ Fraud[{}]를 사용하세요. @@@@@@@@@".format(i))
        if p2 < 0.05:
            logger.info("Normaltest Result: Normal[{}]는 정규 분포가 아닙니다 -- [p-value]: {}".format(i, round(p2, 4)))
            logger.info("********* Normal[{}]는 정규 분포를 만들어야 합니다(log치환 등)! ********".format(i))
        else:
            logger.info("Normal[{}]는 정규 분포입니다! [p-value]: {}".format(i, round(p2, 4)))
            logger.info("@@@@@@@@@ Normal[{}]를 사용하세요. @@@@@@@@@".format(i))
        del stat1, stat2, p1, p2
        
        logger.info("======================== Anderson & Dally test Start ========================")
        stat1 = anderson(fr, dist = 'norm')
        stat2 = anderson(nm, dist = 'norm')
        for k in range(len(stat1.critical_values)):
            sl, cv = stat1.significance_level[k], stat1.critical_values[k]
            if stat1.statistic < stat1.critical_values[k]:
                logger.info("Level[{}]: Fraud[{}]는 정규 분포입니다! [statistic]: {}".format(sl, i, round(stat1.statistic, 4)))
                logger.info("@@@@@@@@@ Fraud[{}]를 사용하세요. @@@@@@@@@".format(i))
            else:
                logger.info("Level[{}] test result : Fraud[{}]는 AD테스트로 인해 정규 분포가 아닙니다 -- [statistic]: {}".format(sl, i, round(stat1.statistic, 4)))
                logger.info("********* Fraud[{}]는 정규 분포를 만들어야 합니다(log치환 등)! *********".format(i))
        for k in range(len(stat2.critical_values)):
            sl, cv = stat2.significance_level[k], stat2.critical_values[k]
            if stat2.statistic < stat2.critical_values[k]:
                logger.info("Level[{}]: Normal[{}]는 정규 분포입니다! [statistic]: {}".format(sl, i, round(stat2.statistic, 4)))
                logger.info("@@@@@@@@@ Normal[{}]를 사용하세요. @@@@@@@@@".format(i))
            else:
                logger.info("Level[{}] test result : Normal[{}]는 AD테스트로 인해 정규 분포가 아닙니다 -- [statistic]: {}".format(sl, i, round(stat2.statistic, 4)))
                logger.info("********* Normal[{}]는 정규 분포를 만들어야 합니다(log치환 등)! *********".format(i))
        del stat1, stat2
        
def cor_test(df, target_col):
    logger = get_logger(os.path.basename(__file__))
    logger.info("[[[[[H0: 독립적입니다.]]]]].")
    logger.info("[[[[[H1: 독립적이지 않습니다.]]]]].")
    
    logger.info("======================== Correlation test Start ========================")
    logger.info("======================== Pearson test Start ========================")
    fr = df[df.F == 1]
    nm = df[df.F == 0]
    for t, k in combinations(target_col, 2):
        logger.info("\n\n======================= columns : [{}, {}] ========================".format(t,k))
        corr1, p1 = pearsonr(fr[t], fr[k])
        if p1 < 0.05:
            logger.info("Pearson Result: Fraud[{}]와 [{}] 는 독립적이지 않습니다. -- [p-value]: {}".format(t,k,round(p1, 4)))
            logger.info("********* Fraud [{}] 와 [{}] 는 둘중 하나를 Drop 또는 교호 작용으로 사용하세요. *********".format(t, k))
        else:
            logger.info("Fraud[{}] 와 [{}] 는 독립적입니다! [p-value]: {}".format(t,k, round(p1, 4)))
            logger.info("@@@@@@@@@ Fraud[{}] 와 [{}] 둘다 사용하세요. @@@@@@@@@".format(t, k))
        corr2, p2 = pearsonr(nm[t], nm[k])
        if p2 < 0.05:
            logger.info("Pearson Result: Normal[{}]와 [{}] 는 독립적이지 않습니다. -- [p-value]: {}".format(t,k,round(p2, 4)))
            logger.info("********* Normal [{}] 와 [{}] 는 둘중 하나를 Drop 또는 교호 작용으로 사용하세요. *********".format(t, k))
        else:
            logger.info("Normal[{}] 와 [{}] 는 독립적입니다! [p-value]: {}".format(t,k, round(p2, 4)))
            logger.info("@@@@@@@@@ Normal[{}] 와 [{}] 둘다 사용하세요. @@@@@@@@@".format(t, k))
    del corr1, corr2, p1, p2
    
    logger.info("======================== Chi-2 test Start ========================")
    for t, k in combinations(target_col, 2):
        logger.info("\n\n======================= columns : [{}, {}] ========================".format(t,k))
        frc = pd.crosstab(fr[t],fr[k])
        stat, p, dof, expected = chi2_contingency(frc)
        if p < 0.05:
            logger.info("Chi-2 Result: Fraud[{}]와 [{}] 는 독립적이지 않습니다. -- [p-value]: {}".format(t,k,round(p, 4)))
            logger.info("********* Fraud [{}] 와 [{}] 는 둘중 하나를 Drop 또는 교호 작용으로 사용하세요. *********".format(t, k))
        else:
            logger.info("Fraud[{}] 와 [{}] 는 독립적입니다! [p-value]: {}".format(t,k, round(p, 4)))
            logger.info("@@@@@@@@@ Fraud[{}] 와 [{}] 둘다 사용하세요. @@@@@@@@@".format(t, k))
        del frc, stat, p, dof, expected
        nmc = pd.crosstab(nm[t],nm[k])
        stat, p, dof, expected = chi2_contingency(nmc)
        if p < 0.05:
            logger.info("Chi-2 Result: Fraud[{}]와 [{}] 는 독립적이지 않습니다. -- [p-value]: {}".format(t,k,round(p, 4)))
            logger.info("********* Fraud [{}] 와 [{}] 는 둘중 하나를 Drop 또는 교호 작용으로 사용하세요. *********".format(t, k))
        else:
            logger.info("Normal[{}] 와 [{}] 는 독립적입니다! [p-value]: {}".format(t,k, round(p, 4)))
            logger.info("@@@@@@@@@ Normal[{}] 와 [{}] 둘다 사용하세요. @@@@@@@@@".format(t, k))
        del nmc, stat, p, dof, expected
    
    
    
    logger.info("======================== Spearman Rank test Start ========================")
    for t, k in combinations(target_col, 2):
        logger.info("\n\n======================= columns : [{}, {}] ========================".format(t,k))
        corr1, p1 = spearmanr(fr[t], fr[k])
        if p1 < 0.05:
            logger.info("Spearman Rank Result: Fraud[{}]와 [{}] 는 독립적이지 않습니다. -- [p-value]: {}".format(t,k,round(p1, 4)))
            logger.info("********* Fraud [{}] 와 [{}] 는 둘중 하나를 Drop 또는 교호 작용으로 사용하세요. *********".format(t, k))
        else:
            logger.info("Fraud[{}] 와 [{}] 는 독립적입니다! [p-value]: {}".format(t,k, round(p1, 4)))
            logger.info("@@@@@@@@@ Fraud[{}] 와 [{}] 둘다 사용하세요. @@@@@@@@@".format(t, k))
        corr2, p2 = spearmanr(nm[t], nm[k])
        if p2 < 0.05:
            logger.info("Spearman Rank Result: Normal[{}]와 [{}] 는 독립적이지 않습니다. -- [p-value]: {}".format(t,k,round(p2, 4)))
            logger.info("********* Normal [{}] 와 [{}] 는 둘중 하나를 Drop 또는 교호 작용으로 사용하세요. *********".format(t, k))
        else:
            logger.info("Normal[{}] 와 [{}] 는 독립적입니다! [p-value]: {}".format(t,k, round(p2, 4)))
            logger.info("@@@@@@@@@ Normal[{}] 와 [{}] 둘다 사용하세요. @@@@@@@@@".format(t, k))
    del corr1, corr2, p1, p2
    
    logger.info("======================== Kendall Rank test Start ========================")
    for t, k in combinations(target_col, 2):
        corr1, p1 = kendalltau(fr[t], fr[k])
        if p1 < 0.05:
            logger.info("Kendall Rank Result: Fraud[{}]와 [{}] 는 독립적이지 않습니다. -- [p-value]: {}".format(t,k,round(p1, 4)))
            logger.info("********* Fraud [{}] 와 [{}] 는 둘중 하나를 Drop 또는 교호 작용으로 사용하세요. *********".format(t, k))
        else:
            logger.info("Fraud[{}] 와 [{}] 는 독립적입니다! [p-value]: {}".format(t,k, round(p1, 4)))
            logger.info("@@@@@@@@@ Fraud[{}] 와 [{}] 둘다 사용하세요. @@@@@@@@@".format(t, k))
        corr2, p2 = kendalltau(nm[t], nm[k])
        if p2 < 0.05:
            logger.info("Kendall Rank Result: Normal[{}]와 [{}] 는 독립적이지 않습니다. -- [p-value]: {}".format(t,k,round(p2, 4)))
            logger.info("********* Normal [{}] 와 [{}] 는 둘중 하나를 Drop 또는 교호 작용으로 사용하세요. *********".format(t, k))
        else:
            logger.info("Normal[{}] 와 [{}] 는 독립적입니다! [p-value]: {}".format(t,k, round(p2, 4)))
            logger.info("@@@@@@@@@ Normal[{}] 와 [{}] 둘다 사용하세요. @@@@@@@@@".format(t, k))
    del corr1, corr2, p1, p2
    
    logger.info("======================== F correlation test Start ========================")
    logger.info("======================== Case 1 ========================")
    for i in target_col:
        logger.info("\n\n======================= columns : [{}] ========================".format(i))
        corr1, p1 = pearsonr(df[i], df["F"])
        if p1 < 0.05:
            logger.info("Pearson : df[{}]는 탐지와 관련이 있습니다! [p-value]: {}".format(i, round(p1, 4)))
            logger.info("@@@@@@@@@ df[{}]를 사용하세요. @@@@@@@@@".format(i))
        else:
            logger.info("Pearson F-cor Result: df[{}]는 탐지와 연관이 없습니다. -- [p-value]: {}".format(i,round(p1, 4)))
            logger.info("********* df[{}]에 대한 검토가 필요합니다! *********".format(i))

    del corr1, p1
    logger.info("======================== Case 2 ========================")
    for i in target_col:
        logger.info("\n\n======================= columns : [{}] ========================".format(i))
        corr1, p1 = spearmanr(df[i], df["F"])
        if p1 < 0.05:
            logger.info("Spearman : df[{}]는 탐지와 관련이 있습니다! [p-value]: {}".format(i, round(p1, 4)))
            logger.info("@@@@@@@@@ df[{}]를 사용하세요. @@@@@@@@@".format(i))
        else:
            logger.info("Spearman F-cor Result: df[{}]는 탐지와 연관이 없습니다. -- [p-value]: {}".format(i,round(p1, 4)))
            logger.info("********* df[{}]에 대한 검토가 필요합니다! *********".format(i))
    del corr1, p1
    
    logger.info("======================== Case 3 ========================")
    for i in target_col:
        logger.info("\n\n======================= columns : [{}] ========================".format(i))
        corr1, p1 = kendalltau(df[i], df["F"])
        if p1 < 0.05:
            logger.info("Kandall : df[{}]는 탐지와 관련이 있습니다! [p-value]: {}".format(i, round(p1, 4)))
            logger.info("@@@@@@@@@ df[{}]를 사용하세요. @@@@@@@@@".format(i))
        else:
            logger.info("Kendall F-cor Result: df[{}]는 탐지와 연관이 없습니다. -- [p-value]: {}".format(i,round(p1, 4)))
            logger.info("********* df[{}]에 대한 검토가 필요합니다! *********".format(i))
    del corr1, p1

def significant_test(df, amt_col):
    logger = get_logger(os.path.basename(__file__))
    logger.info("[[[[[H0: 차이가 없습니다.]]]]]")
    logger.info("[[[[[H1: 차이가 있습니다.]]]]]")
    
    logger.info("======================== Significant test Start ========================")
    logger.info("======================== Parametric test Start ========================")
    
    
    for i in df[amt_col].columns:
        logger.info("\n\n=================================column : {} ====================================".format(i))
        fr = df[df.F == 1]
        nm = df[df.F == 0]
        fr = fr[i]
        nm = nm[i]
        logger.info("======================== Levene and T-test Start ========================")
        stat, p = levene(fr, nm)
        if p < 0.05:
            logger.info("Levene : Fraud, normal [{}]는 등분산 관계가 아닙니다! [p-value]: {}".format(i, round(p, 4)))
            logger.info("@@@@@@@@@ 자동으로 Equal Variable 이 False됩니다")
            change = False
        else:
            logger.info("Levene : Fraud, normal [{}]는 등분산입니다! [p-value]: {}".format(i, round(p, 4)))
            change = True
        del stat, p
        stat, p = ttest_ind(fr, nm, equal_var=change)
        if p < 0.05:
            logger.info("T-test Result: Fraud, Normal[{}]는 차이가 있습니다! -- [p-value]: {}".format(i,round(p, 4)))
            logger.info("@@@@@@@@@ Fraud, Normal[{}]를 사용하세요. @@@@@@@@@".format(i))
        else:
            logger.info("ANOVA Result: Fraud, Normal[{}]는 차이가 없습니다.. [p-value]: {}".format(i, round(p, 4)))
            logger.info("********* [{}]은 검토가 필요합니다! *********".format(i))
        del stat, p
        logger.info("======================== ANOVA Start ========================")
        stat, p = f_oneway(fr, nm)
        if p < 0.05:
            logger.info("ANOVA Result: Fraud, Normal[{}]는 차이가 있습니다! -- [p-value]: {}".format(i,round(p, 4)))
            logger.info("@@@@@@@@@ Fraud, Normal[{}]를 사용하세요. @@@@@@@@@".format(i))
        else:
            logger.info("ANOVA Result: Fraud, Normal[{}]는 차이가 없습니다.. [p-value]: {}".format(i, round(p, 4)))
            logger.info("********* [{}]은 검토가 필요합니다! *********".format(i))
        del stat, p

    logger.info("======================== Non-Parametric test Start ========================")
    for i in df[amt_col].columns:
        logger.info("\n\n=================================column : {} ====================================".format(i))
        fr = df[df.F == 1]
        nm = df[df.F == 0]
        fr = fr[i]
        nm = nm[i]
        logger.info("======================== Mann-WW Start ========================")
        stat, p = mannwhitneyu(fr, nm)
        if p < 0.05:
            logger.info("MWW Result: Fraud, Normal[{}]는 차이가 있습니다! -- [p-value]: {}".format(i,round(p, 4)))
            logger.info("@@@@@@@@@ Fraud, Normal[{}]를 사용하세요. @@@@@@@@@".format(i))
        else:
            logger.info("MWW Result: Fraud, Normal[{}]는 차이가 없습니다.. [p-value]: {}".format(i, round(p, 4)))
            logger.info("********* [{}]은 검토가 필요합니다! *********".format(i))
        del stat, p
        logger.info("======================== Kruskal Start ========================")
        stat, p = kruskal(fr, nm)
        if p < 0.05:
            logger.info("Kruskal Result: Fraud, Normal[{}]는 차이가 있습니다! -- [p-value]: {}".format(i,round(p, 4)))
            logger.info("@@@@@@@@@ Fraud, Normal[{}]를 사용하세요. @@@@@@@@@".format(i))
        else:
            logger.info("Kruskal Result: Fraud, Normal[{}]는 차이가 없습니다.. [p-value]: {}".format(i, round(p, 4)))
            logger.info("********* [{}]은 검토가 필요합니다! *********".format(i))
        del stat, p


"""
XAI
https://github.com/statsmodels/statsmodels/blob/master/examples/python/gls.py
https://codeday.me/ko/qa/20190322/54104.html
https://datascienceschool.net/view-notebook/58269d7f52bd49879965cdc4721da42d/
https://www.statsmodels.org/dev/examples/notebooks/generated/gls.html
http://blog.naver.com/PostView.nhn?blogId=jieun0441&logNo=221133814810&parentCategoryNo=&categoryNo=30&viewDate=&isShowPopularPosts=false&from=postView
http://blog.naver.com/PostView.nhn?blogId=yonxman&logNo=220950614789&categoryNo=1&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=search
"""