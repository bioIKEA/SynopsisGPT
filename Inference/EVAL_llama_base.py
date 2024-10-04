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

from transformers import pipeline, AutoModel
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import TFAutoModelForQuestionAnswering
from transformers import create_optimizer
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append('../')


#-- Internal Imports --#
from APIs.Utils.Utils import Utils
from APIs.DataPrep.CCR_DataPrep import CCR_DataPrep
from APIs.Metrics.LLM_Metrics import LLM_Metrics
#---------------------------------------------------#

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#-- Code for Evaluating llama models --#

#--Instantiate class objects--#
UtilObj = Utils()
DPObj = CCR_DataPrep()
METObj = LLM_Metrics()

#%%

reprocess_data = 0
get_new_qdf = 0

#-- Separate data quesiton, context, answers --#
max_input_size = 3795

data_sel = ''
# data_sel = '_cl'
# data_sel = '_rcl'
# data_sel = '_qa'
# data_sel = '_gen'

fine_tuned = 0

#-- Get question_df --#
if(get_new_qdf == 1):    
    question_df = DPObj.get_question_df()   
else:
    load_path = "DataPrep/"
    load_name = "qa_data_size_"+str(max_input_size) + str(data_sel)
        
    question_df = np.load(load_path + load_name + '_question_df.pkl', allow_pickle = True)

if(reprocess_data == 1):  
    # Option for preparing the data again
    #--Loading synoptic reports data --#
    load_path = "../../Imp/Data/ProcessedData/"
    load_name = 'uniq_cancer'
    
    #--Train-val data --#
    # raw_train_val_df = np.load(load_path + load_name + '_train_val_df.pkl', allow_pickle = True)
    #-- Test data --#
    raw_test_df = np.load(load_path + load_name +'_test1_df.pkl', allow_pickle = True)
    
    # qa_data_train_val_df = DPObj.convert_to_qa_df(raw_train_val_df, question_df, max_input_size)
    qa_data_test_df = DPObj.convert_to_qa_df(raw_test_df, question_df, max_input_size)

else:
    # For loading previously prepared data
    load_path = "DataPrep/"
    load_name = "qa_data_size_"+str(max_input_size) + str(data_sel)
    
    
    # qa_data_train_val_df = np.load(load_path + load_name + '_train_val_df.pkl', allow_pickle = True)
    qa_data_test_df = np.load(load_path + load_name + '_test1_df.pkl', allow_pickle = True)

#-- End of special operations for cancer data --#





#%%
# Llama 2 model
# import transformers
import torch

model_str = "meta-llama/Meta-Llama-3-8B"
# model_str = "meta-llama/Llama-2-7b-hf"
# model_str = "meta-llama/Llama-2-13b-hf"
# model_str = "meta-llama/Llama-2-70b-hf"

# Check and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(fine_tuned == 0):
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    
    # Load fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(model_str)
    model = model.to(device)

    model_name = model_str.replace('/','_') + '0shot_rawref_'+str(max_input_size)+'v_trial'
    
else:
    model_name = 'meta-llama_Llama-2-7b-hf_FT_3795tokens_-1steps_MainGenFull_v1'
    path_to_model = 'TrainedModels/llama2/Fine_tuned/' + model_name + '/'
    
    # Load pre-trained model
    # model = AutoModelForCausalLM.from_pretrained(path_to_model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    
    # Load fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(path_to_model)
    model = model.to(device)


#%%
# Modified function for qa cancer clinical notes dataset
def create_test_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters 
    :param sample: Sample dictionnary
    """
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "### Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['question']}"
    input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
    response = f"{RESPONSE_KEY}"#"\n{sample['nlp_raw_ref']}"
    end = ''#f"{END_KEY}"
    
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    formatted_prompt = "\n\n".join(parts)
    
    sample["text"] = formatted_prompt

    return sample


#%%
#-- Evaluate results on the test data --#
preds = pd.DataFrame([])

preds['DOC_ID'] = []
preds['QUESTION'] = []
preds['CONTEXT'] = []
preds['NLP_RAW_REFERENCE'] = []
preds['NLP_REFERENCE'] = []
preds['MODEL_OUT'] = []
preds['MANUAL_CHK'] = []



raw_test_df = qa_data_test_df.reset_index(drop = True)
lcl_cnt = 0 

N_test_files = 100 #Number of unique files to test, use -1 for testing all

for jj in range(0,np.size(question_df,0),1):
    question = question_df.loc[jj,'question']
    max_new_tokens = 30#question_df.loc[jj,'max_new_tokens'] 

    qa_chk = raw_test_df.loc[:,'question'] == question
    sub_raw_test_df = raw_test_df.loc[qa_chk,:].reset_index(drop = True)
    sub_raw_size = np.size(sub_raw_test_df,0)
    
    sub_n_chk = pd.Series(np.zeros(sub_raw_size))
    if(N_test_files == -1):
        sub_list = np.unique(sub_raw_test_df.loc[:,'doc_id'])
    else:
        sub_list = np.unique(sub_raw_test_df.loc[:,'doc_id'])[0:N_test_files]    
    
    for kk in range(0,sub_raw_size,1):
        
        if(sub_raw_test_df.loc[kk, 'doc_id'] in sub_list):
            sub_n_chk[kk] = True
        else:
            sub_n_chk[kk] = False
    
    sub_n_raw_test_df = sub_raw_test_df.loc[sub_n_chk,:].reset_index(drop = True)
    N_sub_elements = np.size(sub_n_raw_test_df,0)    
    
    
    for ii in tqdm(range(0,N_sub_elements,1)):
                
        try:
            context = sub_n_raw_test_df.loc[ii,'context'].replace('\r\n  ','')
            syn_report = sub_n_raw_test_df.loc[ii,'answers']['text']               
            
            
            #-- Predict the answer --#            
            #Zero shot
            sample_in = sub_n_raw_test_df.loc[ii,:]
            sample_out = create_test_prompt_formats(sample_in)
            prompt = sample_out['text']
            
            
            context_len = len(prompt)
            

            # Tokenize input text
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Get answer
            # (Adjust max_new_tokens variable as you wish (maximum number of tokens the model can generate to answer the input))
            outputs = model.generate(input_ids=inputs["input_ids"], 
                                      attention_mask=inputs["attention_mask"], 
                                      max_new_tokens=max_new_tokens, 
                                      pad_token_id=tokenizer.eos_token_id)
            
            
            # Decode output & print it
            gen_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)        
        
        
        
            # Save answers
            preds.loc[lcl_cnt,'DOC_ID'] = sub_n_raw_test_df.loc[ii,'doc_id']
            preds.loc[lcl_cnt,'QUESTION'] = question
            preds.loc[lcl_cnt,'CONTEXT'] = prompt
            preds.loc[lcl_cnt,'NLP_RAW_REFERENCE'] = sub_n_raw_test_df.loc[ii,'nlp_raw_ref']
            preds.loc[lcl_cnt, 'NLP_REFERENCE'] = syn_report[0]
            preds.loc[lcl_cnt, 'MODEL_OUT'] = str(gen_sequence[context_len::])
            preds.loc[lcl_cnt, 'MANUAL_CHK'] = 0
            preds.loc[lcl_cnt,'JARO_SIMILARITY'] = jellyfish.jaro_similarity(preds.loc[lcl_cnt, 'MODEL_OUT'], preds.loc[lcl_cnt,'NLP_REFERENCE'])
            preds.loc[lcl_cnt,'JACCARD_SIMILARITY'] = METObj.jaccard_similarity(preds.loc[lcl_cnt, 'MODEL_OUT'], preds.loc[lcl_cnt,'NLP_REFERENCE'])
                
        
        
            lcl_cnt = lcl_cnt + 1
        
        except:
            print('Error encountered, skipping file')

#%%
preds.to_pickle('Results/'+model_name+'_'+str(N_test_files)+'_preds_v1'+'.pkl')
question_df.to_pickle('Results/'+model_name+'_'+str(N_test_files)+'_qdf_v1'+'.pkl')
preds.to_csv(path_or_buf = 'Results/'+model_name+'_'+str(N_test_files)+'_PREDS_v1.csv')

