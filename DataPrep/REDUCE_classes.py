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

import tensorflow as tf

# import datasets
import jellyfish

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import TFAutoModelForQuestionAnswering
from transformers import create_optimizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM

import sys
sys.path.append('../')

#-- Internal Imports --#
from APIs.Utils.Utils import Utils
from APIs.DataPrep.CCR_DataPrep import CCR_DataPrep
from APIs.Metrics.LLM_Metrics import LLM_Metrics
#---------------------------------------------------#



# Reduce the number of classes 



#--Instantiate class objects--#
UtilObj = Utils()
DPObj = CCR_DataPrep()
METObj = LLM_Metrics()

#%%

reprocess_data = 1
get_new_qdf = 1

#-- Separate data quesiton, context, answers --#
max_input_size = 1400


# data_sel = ''
# data_sel = '_cl'
# data_sel = '_qa'
# data_sel = '_gen'
data_sel = '_MTCNN_cl'


#-- Get question_df --#
if(get_new_qdf == 1):    
    question_df = DPObj.get_question_df()   
else:
    load_path = "DataPrep/ProcessedDataPkl/"
    load_name = "qa_data_size_"+str(max_input_size) + str(data_sel)
        
    question_df = np.load(load_path + load_name + '_question_df.pkl', allow_pickle = True)

if(reprocess_data == 1):  
    # Option for preparing the data again
    #--Loading synoptic reports data --#
    load_path = "../../Imp/Data/ProcessedData/"
    load_name = 'uniq_cancer'
    
    #--Train-val data --#
    raw_train_val_df = np.load(load_path + load_name + '_train_val_df.pkl', allow_pickle = True)
    #-- Test data --#
    raw_test1_df = np.load(load_path + load_name +'_test1_df.pkl', allow_pickle = True)
    raw_test2_df = np.load(load_path + load_name +'_test2_df.pkl', allow_pickle = True)
    
    qa_data_train_val_df = DPObj.convert_to_qa_df(raw_train_val_df, question_df, max_input_size)
    qa_data_test_df = DPObj.convert_to_qa_df(raw_test1_df, question_df, max_input_size)
    qa_data_test2_df = DPObj.convert_to_qa_df(raw_test2_df, question_df, max_input_size)
else:
    # For loading previously prepared data
    load_path = "DataPrep/ProcessedDataPkl/"
    load_name = "qa_data_size_"+str(max_input_size)+ str(data_sel)
    
    
    qa_data_train_val_df = np.load(load_path + load_name + '_train_val_df.pkl', allow_pickle = True)
    qa_data_test1_df = np.load(load_path + load_name + '_test1_df.pkl', allow_pickle = True)
    # qa_data_test2_df = np.load(load_path + load_name + '_test2_df.pkl', allow_pickle = True)

#-- End of special operations for cancer data --#



#%%

    

def get_rcl(qa_data):

    reduced_cl_df = np.load('DataPrep/reduced_cl_df.pkl', allow_pickle = True)
    question_dict = dict(zip(question_df['dict_ref'], question_df['question']))
    
    # qa_data.loc[:, 'nlp_raw_ref_old'] = qa_data.loc[:, 'nlp_raw_ref']
    
    for ii in tqdm(range(0,np.size(reduced_cl_df,0),1)):
    
        q_header = reduced_cl_df.loc[ii,'dict_ref']
        question = question_dict[reduced_cl_df.loc[ii,'dict_ref']]
    
        rcl_dict = reduced_cl_df.loc[ii,'rcl_dict'][0]
    
    
        for jj in tqdm(range(0,np.size(qa_data,0),1)):
            
            try:
                if(qa_data.loc[jj,'question'] == question):
                    
                    qa_data.loc[jj, 'nlp_raw_ref'] = rcl_dict[qa_data.loc[jj, 'nlp_raw_ref']]
                  
            except:
                print('skipping to next variable')
        


    return(qa_data)







#%%

save_path = load_path
save_name = "qa_data_size_"+str(max_input_size)+ '_rcl'

if(data_sel == '_cl'):

    qa_data_train_val_df = get_rcl(qa_data_train_val_df)
    qa_data_test1_df = get_rcl(qa_data_test1_df)


    question_df.to_pickle(save_path + save_name + '_question_df.pkl')
    qa_data_train_val_df.to_pickle(save_path + save_name + '_train_val_df.pkl')
    qa_data_test1_df.to_pickle(save_path + save_name + '_test1_df.pkl')



    
    
    











