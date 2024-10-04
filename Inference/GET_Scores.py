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
# import jellyfish
# import evaluate
from evaluate import load

# from transformers import pipeline

import sys
sys.path.append('../')


#-- Internal Imports --#
from APIs.Utils.Utils import Utils
from APIs.DataPrep.CCR_DataPrep import CCR_DataPrep
from APIs.Metrics.LLM_Metrics import LLM_Metrics
#---------------------------------------------------#

#%%
#--Instantiate class objects--#
UtilObj = Utils()
DPObj = CCR_DataPrep()
METObj = LLM_Metrics()


# master_df = pd.read_csv('GET_Scores_Master.csv')
master_df = pd.read_excel('GET_Scores_Master.xlsx')
master_df = master_df.dropna(axis = 0)
master_df = master_df.reset_index(drop = True)

# compute bert scores
bertscore = load("bertscore", device = 'cuda')
# score_model = 'distilbert-base-uncased'
score_model = 'roberta-large-mnli'


#%%
for mst_cnt in tqdm(range(0,np.size(master_df,0),1), desc = 'Outer Loop'):

    # Check if scores need to be generated
    get_scores = master_df.loc[mst_cnt,'get_scores']
    
    if(get_scores == 1):
    
        preds_filename = master_df.loc[mst_cnt,'preds_pickle']
        qdf_filename = master_df.loc[mst_cnt,'QDF_pickle'] 
        file_folder_path = master_df.loc[mst_cnt,'folder_path'] 
        
        
        # Check if model output needs cleaning
        CLEAN_MODEL_OUT = master_df.loc[mst_cnt,'clean_model_out']
        
        # Load result file and question_df file
        preds = np.load('Results/'+file_folder_path+preds_filename+'.pkl', allow_pickle = True)
        question_df = np.load('Results/'+file_folder_path+qdf_filename+'.pkl', allow_pickle = True)
    
    
        if(CLEAN_MODEL_OUT == 1):
        
            preds['RAW_MODEL_OUT'] = ''
            
            for ii in range(0,np.size(preds,0),1):
                
                old_m_out = preds.loc[ii,'MODEL_OUT']
                
                end_loc = old_m_out.find("\n\n### End") 
                
                if(end_loc == -1):
                    new_m_out = old_m_out
                else:
                    new_m_out = old_m_out[0:end_loc]
                
                preds.loc[ii,'RAW_MODEL_OUT'] = old_m_out
                preds.loc[ii,'MODEL_OUT'] = new_m_out
            
        
    
    
        for ii in tqdm(range(0,np.size(preds,0),1)):
            try:
                m_out = preds.loc[ii,'MODEL_OUT']
                m_ref = preds.loc[ii,'NLP_RAW_REFERENCE']
                results = bertscore.compute(predictions=[m_out], references=[m_ref], lang="en", model_type = score_model)
                preds.loc[ii,'BERTScore_F1'] = results['f1'][0]
                preds.loc[ii,'BERTScore_PRES'] = results['precision'][0]
                preds.loc[ii,'BERTScore_RECALL'] = results['recall'][0]
                preds.loc[ii,'BERTScore_Model'] = score_model
            except:
                preds.loc[ii,'BERTScore_F1'] = 0
                preds.loc[ii,'BERTScore_PRES'] = 0
                preds.loc[ii,'BERTScore_RECALL'] = 0     
                preds.loc[ii,'BERTScore_Model'] = score_model
                print('Error, moving to next item')
        
    
    # Save the new calculated metrics to file
    preds.to_pickle('Results/'+file_folder_path+preds_filename+'_scores.pkl')









