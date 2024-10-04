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
# from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
# from transformers import TFAutoModelForQuestionAnswering
from transformers import create_optimizer
from transformers import DataCollatorForLanguageModeling
# from transformers import AutoModelForMaskedLM
from transformers import AutoModelForCausalLM

import sys
sys.path.append('../')

#-- Internal Imports --#
from APIs.Utils.Utils import Utils
from APIs.DataPrep.CCR_DataPrep import CCR_DataPrep
from APIs.Metrics.LLM_Metrics import LLM_Metrics
#---------------------------------------------------#

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

#-- Code for evaluating fine tuned qa models --#


#--Instantiate class objects--#
UtilObj = Utils()
DPObj = CCR_DataPrep()
METObj = LLM_Metrics()

#%%
reprocess_data = 0
get_new_qdf = 0

#-- Separate data quesiton, context, answers --#
max_input_size = 385

# data_sel = ''
# data_sel = '_cl'
# data_sel = '_qa'
data_sel = '_gen'
N_steps = -1

fine_tuned = 1
ft_model_add_str = '_FT_SPGEN_v2'
pre_model_add_str = '_'+str(max_input_size)+'_PreT_Main_v2'

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
    
    
    qa_data_train_val_df = np.load(load_path + load_name + '_train_val_df.pkl', allow_pickle = True)
    qa_data_test_df = np.load(load_path + load_name + '_test1_df.pkl', allow_pickle = True)

#-- End of special operations for cancer data --#

#%%

#-- Load fine-tuned or pretrained model --#
# from transformers import AutoModelForCausalLM

# model_str = "bert-base-uncased" #"distilbert-base-cased-distilled-squad" #"distilbert-base-uncased"
# model_str =  "distilbert-base-cased-distilled-squad" 
# model_str = "EleutherAI/gpt-j-6B"
# model_str = "facebook/bart-base"
# model_str = "facebook/bart-large-cnn"
# model_str  = "gpt2"
model_str = "bert-large-uncased"
# model_str = "bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(model_str)
# MASK_TOKEN = tokenizer.mask_token


# tokenizer.pad_token = tokenizer.eos_token

if(fine_tuned == 1):
    model_name = model_str +'_FT_'+str(max_input_size)+'tokens_'+\
        str(data_sel)+str(N_steps)+ ft_model_add_str
    ft_model = AutoModelForCausalLM.from_pretrained("TrainedModels/CLM/"+model_name)#("TrainedModels/"+str(model_name))    
    model = ft_model
    text_generator = pipeline("text-generation", model=model, tokenizer = tokenizer, device_map = 'cuda:0')
else:
    model_name = model_str + pre_model_add_str
    pre_model = AutoModelForCausalLM.from_pretrained(model_str)
    model = pre_model 
    text_generator = pipeline("text-generation", model=model, tokenizer = tokenizer)

model.to('cuda:0')
# tokenizer.to('cuda')
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

# preds['NOTE_NUMBER'] = []
preds['DOC_ID'] = []
preds['QUESTION'] = []
preds['CONTEXT'] = []
preds['NLP_RAW_REFERENCE'] = []
preds['NLP_REFERENCE'] = []
preds['MODEL_OUT'] = []
preds['MANUAL_CHK'] = []
# preds['alternate'] = []

raw_test_df = qa_data_test_df.reset_index(drop = True)
lcl_cnt = 0 

N_test_files = 100 #Number of unique files to test
N_samples = np.size(raw_test_df,0)

for jj in range(0,np.size(question_df,0),1):
    
    question = question_df.loc[jj,'question']
    qa_chk = raw_test_df.loc[:,'question'] == question
    sub_raw_test_df = raw_test_df.loc[qa_chk,:].reset_index(drop = True)
    sub_raw_size = np.size(sub_raw_test_df,0)

    sub_n_chk = pd.Series(np.zeros(sub_raw_size))
    sub_list = np.unique(sub_raw_test_df.loc[:,'doc_id'])[0:N_test_files]
    
    
    for kk in range(0,sub_raw_size,1):
        
        if(sub_raw_test_df.loc[kk, 'doc_id'] in sub_list):
            sub_n_chk[kk] = True
        else:
            sub_n_chk[kk] = False
    
    sub_n_raw_test_df = sub_raw_test_df.loc[sub_n_chk,:].reset_index(drop = True)
    N_sub_elements = np.size(sub_n_raw_test_df,0)

    for ii in tqdm(range(0,N_sub_elements,1)):
          
        context = sub_n_raw_test_df.loc[ii,'context'].replace('\r\n  ','')
        syn_report = sub_n_raw_test_df.loc[ii,'answers']['text']   

        # try:
        #-- Predict the answer --#
        #Dec 11
        #input_text = sub_n_raw_test_df.loc[ii,'context'] +'\n\n ### Instruction: ' + sub_n_raw_test_df.loc[ii,'question'] + '\n\n ### Context: '+ '\n\n ### Response: ' 
               
        input_text = create_test_prompt_formats(sub_n_raw_test_df.loc[ii,:])
        input_text = input_text.loc['text']
        
        
        inputs = tokenizer(input_text, return_tensors = 'pt').to('cuda:0')
        
        output = model.generate(**inputs, max_new_tokens = 30)
        
        predicted_answer = tokenizer.decode(output[0], skip_special_tokens = True)
        
        
        #-- Alternative using pipeline --#
        
        
        # input_text = "Your input text here"
        '''
        output = text_generator(input_text, max_new_tokens=15)  # Adjust max_length as needed
        predicted_answer = output[0]['generated_text']
        '''
        # print(predicted_answer)
        
        
        # Save answers
        preds.loc[lcl_cnt,'DOC_ID'] = sub_n_raw_test_df.loc[ii,'doc_id']
        preds.loc[lcl_cnt,'QUESTION'] = question
        preds.loc[lcl_cnt,'CONTEXT'] = input_text#context
        preds.loc[lcl_cnt,'NLP_RAW_REFERENCE'] = sub_n_raw_test_df.loc[ii,'nlp_raw_ref']
        preds.loc[lcl_cnt, 'NLP_REFERENCE'] = syn_report[0]#syn_report[question_df.loc[jj,'dict_ref']]
        preds.loc[lcl_cnt, 'MODEL_OUT'] = predicted_answer[len(input_text)::]
        preds.loc[lcl_cnt, 'MANUAL_CHK'] = 0
        preds.loc[lcl_cnt,'JARO_SIMILARITY'] = jellyfish.jaro_similarity(preds.loc[lcl_cnt, 'MODEL_OUT'], preds.loc[lcl_cnt,'NLP_RAW_REFERENCE'])
        preds.loc[lcl_cnt,'JACCARD_SIMILARITY'] = METObj.jaccard_similarity(preds.loc[lcl_cnt, 'MODEL_OUT'], preds.loc[lcl_cnt,'NLP_RAW_REFERENCE'])
        # preds.loc[lcl_cnt, 'alternate'] = context[answer_start_index:answer_end_index]
    
        lcl_cnt = lcl_cnt + 1
        
        # except:
        #     print('Error encountered, skipping file')

#%%
save_model_name = model_name.replace('/','_')
preds.to_pickle('Results/'+save_model_name+'_'+str(N_test_files)+'_preds'+'.pkl')
question_df.to_pickle('Results/'+save_model_name+'_'+str(N_test_files)+'_qdf'+'.pkl')
preds.to_csv(path_or_buf = 'Results/'+save_model_name+'_'+str(N_test_files)+'_PREDS.csv')

    