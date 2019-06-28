#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:22:01 2019

@author: musaidawan

Goal: To get data for subgroup analysis 


"""


import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split # use to split data
import os

# home directory 
os.chdir("/Users/musaidawan/Dropbox/Duke/Projects/IV FLAME/Application/5 mins vote/replication_with_flame")

#read in data
df = pd.read_stata('analysis.dta')
df = df.reset_index()

#select full control list

full_controls = ["territory", "prop_leftabstention_an", "nb_registered_mun", "prop_turnout_pr07t1_an",  
                 "nb_registered_pr12t1_an", "population", "share_men", "share_0014", "share_1529", 
                 "share_3044", "share_4559", "share_6074","share_75p", "share_unemployed", 
                 "share_working_pop", "population_delta", "share_men_delta", "share_0014_delta","share_1529_delta", 
                 "share_3044_delta", "share_4559_delta", "share_6074_delta","share_75p_delta","share_unemployed_delta", 
                 "share_working_pop_delta"]

#--- var for subgroup analysis
sub_group = "share_men" # options: share_men, median_income

if sub_group == "share_men" :
    var_subgroup = ["share_men_sub"]
    df[var_subgroup[0]] = pd.cut(df[sub_group], bins = [.26, 0.4999, 0.8 ], labels = ["Female in Majority", "Male in Majority"] ) # create 5 bins for each       
    #full_controls = list(set(full_controls) - set(var_subgroup))
    groups = ["Female in Majority", "Male in Majority"]

if sub_group == "median_income" :    
    var_subgroup = ["median_income_sub"]
    df[var_subgroup[0]] = pd.cut(df[sub_group], bins = [10312, 16789,  21035, 23829], labels = ["Low", "Medium","High"] ) # create 5 bins for each       
    full_controls = full_controls 
    groups = ["Low", "Medium","High"]
    
    
if sub_group == "cities" :
    df["Marseille"] = df["territory"].str.contains('Marseille', regex=False)
    df["Paris"] = df["territory"].str.contains('Paris', regex=False)
    df["Lyon"] = df["territory"].str.contains('Lyon', regex=False)
    groups = ["Marseille", "Paris","Lyon"]
    

#--- end of subgroup analysis

outcomes =["prop_turnout_pr12t1_an", "prop_turnout_pr12t2_an", "prop_turnout_pr12t12_an", 
           "prop_hollande_pr12t1_an", "prop_hollande_pr12t2_an","prop_hollande_pr12t12_an"]


# remove rows with missing values of outcome
for outcome in outcomes:
		df = df.loc[~np.isnan(df[outcome])] 
            # remove rows for which any of the outcome is missing!

df = df.loc[~np.isnan(df["prop_turnout_pr07t1_an"])]  # dropping missing values of prop_turnout_pr07t1_an
print(df[full_controls].isnull().any())  # no col is missing!

# discretize continuous variables            
for cov in full_controls: # bin cont covs
    if cov in ["territory"] :
        continue
    if cov in ["prop_leftabstention_an"] :
        df[cov] = pd.cut(df[cov], bins = 10 , labels = ["1", "2", "3", "4","5","6","7","8","9","10"] ) # create 5 bins for each    
    else :
        df[cov] = pd.cut(df[cov], bins = 5 , labels = ["1", "2", "3", "4","5"] ) # create 5 bins for each


# rename columns for iv and treated
df.rename(index=str, columns={"treatment": "iv", "allocated": "treated"},inplace=True)

# adding strata col, not treating as a covariate so no renaming is needed
#df_strat = df[ ["index","stratum_identifier"] ]

#select and save data
for outcome in outcomes :
    for subgroup in groups :
        
        # selection
        if sub_group == "cities":
            df_sub = df[ df[subgroup] == True ]
        else :
            df_sub = df[ df[var_subgroup[0]] == subgroup ]

        # projection        
        df_cov = df_sub[["index"]+full_controls]

        # rename
        i_t_y = ["index","iv","treated",outcome]
        df_i_t_y = df[i_t_y]
        df_i_t_y.columns = ["index","iv","treated","outcome"]
        
        
        
        # rename covariates in df_cov
        q = np.size(df_cov, 1) # number of feature columns
        ls = range( q  - 1) # -1 for index col
        ls = map(str, ls) 
        
        # dictionary for renames 
        covs = list(df_cov.columns)
        covs = covs[1:]
        dictionary = dict(zip(ls, covs))
        
        df_cov.columns = ["index"] + ls
    
        # df_train 
        df_train_0 = pd.merge(df_cov,df_i_t_y,on='index',how='inner')
        #df_train = pd.merge(df_train,df_strat,on='index',how='inner')
        train, test = train_test_split(df_train_0, test_size=0.15, random_state = 123, stratify =df_train_0[["iv","treated"]] )
    
        # df_hold (random sample of df_train)
        #size  = int(np.size(df_train,0)*0.9)
        #df_hold = df_train.sample(n=size, random_state=2019)
    
        # check for missing values
        df_train = train
        df_hold = test        
        print(df_train.isnull().any())
        print(df_hold.isnull().any())
        
        pickle.dump(df_train, open('data/subgroup/' +subgroup+ '/train_'+ outcome + subgroup, 'wb')) 
        pickle.dump(df_hold, open('data/subgroup/' +subgroup+ '/hold_'+ outcome + subgroup, 'wb')) 
        pickle.dump(dictionary, open('data/subgroup/' +subgroup+ '/dictionary', 'wb')) 
        

