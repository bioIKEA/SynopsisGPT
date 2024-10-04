# -*- coding: utf-8 -*-
"""
WARNING: This code is intended for research and development purposes only. 
It is not intended for clinical or medical use. It has not been reviewed or 
approved by any medical or regulatory authorities. 
"""

# =============================================================================
# Description
# =============================================================================

# Call script for preparing the cancer clinical  reports for analysis by LLMs

# =============================================================================
 
# =============================================================================
# Imports
# ============================================================================= 
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd


import sys
sys.path.append('../../')

# Internal Imports
from APIs.Utils.Utils import Utils
from APIs.DataPrep.CCR_DataPrep import CCR_DataPrep
# =============================================================================


UtilObj = Utils()
DPObj = CCR_DataPrep()


#--Loading synoptic reports data --#
uniq_cancer_data_df = np.load('../../../Imp/Data/uniq_cancer_data_df.pkl', allow_pickle = True)
uniq_cancer_data_df = uniq_cancer_data_df.reset_index(drop = True)

data = uniq_cancer_data_df


#%%
#--Get all unique headers from synoptic reports for cleanup--#
data_path = '../../../Imp/Data/ProcessedData/unique_keys.csv'
uniq_keys_dict = DPObj.get_uniq_headers(data, data_path = data_path)


#--Clean up headers of the synoptic reports in the raw data using the manually --#
#--cleaned csv files (eliminates typos and redundancies for important headers) --#
clean_dict_df = pd.read_csv('../../../Imp/Data/ProcessedData/unique_keys_mod.csv')
clean_dict = dict(clean_dict_df.iloc[:,0:2].values)

# Data with clean headers for variables of interest
data = DPObj.get_clean_headers(data, clean_dict) 


clean_data_path = '../../../Imp/Data/ProcessedData/clean_unique_keys.csv'
uniq_keys_clean_dict = DPObj.get_uniq_headers(data, data_path = clean_data_path)




#%%
#--Split and save train-val-test data--#
seed = 9
train_val_per = 0.8

save_path = "../../../Imp/Data/ProcessedData/"
save_name = 'uniq_cancer'

[raw_train_val_df, raw_test_df] = DPObj.split_data(data, train_val_per = train_val_per, seed = seed)


train_val_per = 0.5
[raw_test1_df, raw_test2_df] = DPObj.split_data(raw_test_df, train_val_per = train_val_per, seed = seed)


UtilObj.folder_check(save_path)
raw_train_val_df.to_pickle(save_path + save_name + '_train_val_df.pkl')
raw_test1_df.to_pickle(save_path + save_name + '_test1_df.pkl')
raw_test2_df.to_pickle(save_path + save_name + '_test2_df.pkl')




