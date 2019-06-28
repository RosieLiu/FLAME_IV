#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 22:17:09 2019

@author: musaidawan

Goals : To study some examples of matched groups

"""

import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split # use to split data
import os

# home directory 
#os.chdir("/Users/musaidawan/Dropbox/Duke/Projects/IV FLAME/Application/5 mins vote/replication_with_flame")
os.chdir("/Users/marco/Dropbox/Duke/projects/flame_iv/replication_with_flame")

#read in data
df = pd.read_stata('analysis.dta') # original data set
df = df.reset_index()

# matched data
df2 = pickle.load(open('data/compare_groups/matched_train_prop_hollande_pr12t1_an' , "rb"))
df2 = df2[["index",'group_id_overall']]


#select full control list
full_controls = ["territory", "prop_leftabstention_an", "nb_registered_mun", "prop_turnout_pr07t1_an",  
                 "nb_registered_pr12t1_an", "population", "share_men", "share_0014", "share_1529", 
                 "share_3044", "share_4559", "share_6074","share_75p", "share_unemployed", 
                 "share_working_pop", "population_delta", "share_men_delta", "share_0014_delta","share_1529_delta", 
                 "share_3044_delta", "share_4559_delta", "share_6074_delta","share_75p_delta","share_unemployed_delta", 
                 "share_working_pop_delta"]

#full_controls = full_controls[1:]

outcomes =["prop_turnout_pr12t1_an", "prop_turnout_pr12t2_an", "prop_turnout_pr12t12_an", 
           "prop_hollande_pr12t1_an", "prop_hollande_pr12t2_an","prop_hollande_pr12t12_an"]

# remove rows with missing values of outcome
for outcome in outcomes:
		df = df.loc[~np.isnan(df[outcome])] 

# discretize continuous variables            
for cov in full_controls: # bin cont covs
    if cov in ["territory"] :
        continue
    if cov in ["prop_leftabstention_an"] :
        df[cov] = pd.cut(df[cov], bins = 10, precision=2) 
#labels = ["1", "2", "3", "4","5","6","7","8","9","10"] ) # create 5 bins for each    
    else :
        df[cov] = pd.cut(df[cov], bins = 5, precision=2)
#labels = ["1", "2", "3", "4","5"] ) # create 5 bins for each


# merged data 
df_merged = pd.merge(df,df2,on='index',how='inner')
df_merged.sort_values(['group_id_overall'],ascending=True, inplace=True)


rep_covs = ["group_id_overall", "territory", "prop_leftabstention_an",
            "prop_turnout_pr07t1_an", "population", "share_men",
            "share_unemployed", "allocated", "treatment"]

# can select set of covariates we are interested in
df_view = df_merged[ rep_covs ]


print df_view.loc[df_view["group_id_overall"] == 248, rep_covs[1:]].to_latex(index=False)
print df_view.loc[df_view["group_id_overall"] == 282, rep_covs[1:]].to_latex(index=False)

# 246 Small villages

# 79  City of Nantes
# 100 St. Nazaire
# 248
# 282
