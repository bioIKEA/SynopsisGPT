# -*- coding: utf-8 -*-
"""
WARNING: This code is intended for research and development purposes only. 
It is not intended for clinical or medical use. It has not been reviewed or 
approved by any medical or regulatory authorities. 
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import os as os
from tqdm import tqdm

import datasets
import tensorflow as tf

import jellyfish

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import TFAutoModelForQuestionAnswering
from transformers import create_optimizer

import sys
sys.path.append('../')

#-- Internal Imports --#
from APIs.Utils.Utils import Utils
from APIs.DataPrep.CCR_DataPrep import CCR_DataPrep
#---------------------------------------------------#
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#-- Code for fine tuneing qa models --#


#--Instantiate class objects--#.
UtilObj = Utils()
DPObj = CCR_DataPrep()





#%%
#--Loading synoptic reports data --#
load_path = "../../Imp/Data/ProcessedData/"
load_name = 'uniq_cancer'

save_path = "DataPrep/"

#--Train-val data --#
raw_train_val_df = np.load(load_path + load_name + '_train_val_df.pkl', allow_pickle = True)
#-- Test 1 data --#
raw_test1_df = np.load(load_path + load_name +'_test1_df.pkl', allow_pickle = True)
#-- Test 2 data --#
raw_test2_df = np.load(load_path + load_name +'_test2_df.pkl', allow_pickle = True)

#-- Get question_df --#
question_df = DPObj.get_question_df()

#-- Separate data quesiton, context, answers --#
max_input_size = 500
# data_sel = ''
data_sel = '_cl'
# data_sel = '_qa'
# data_sel = '_gen'


qa_data_train_val_df = DPObj.convert_to_qa_df(raw_train_val_df, question_df, max_input_size)
qa_data_test1_df = DPObj.convert_to_qa_df(raw_test1_df, question_df, max_input_size)
qa_data_test2_df = DPObj.convert_to_qa_df(raw_test2_df, question_df, max_input_size)

save_name = 'qa_data_size_'+str(max_input_size) + str(data_sel)

question_df.to_pickle(save_path + save_name + '_question_df.pkl')
qa_data_train_val_df.to_pickle(save_path + save_name + '_train_val_df.pkl')
qa_data_test1_df.to_pickle(save_path + save_name + '_test1_df.pkl')
qa_data_test2_df.to_pickle(save_path + save_name + '_test2_df.pkl')


#%%

    
    
    
    
    
    
    


