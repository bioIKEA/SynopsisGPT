# -*- coding: utf-8 -*-
"""
WARNING: This code is intended for research and development purposes only. 
It is not intended for clinical or medical use. It has not been reviewed or 
approved by any medical or regulatory authorities. 
"""

# =============================================================================
# Description
# =============================================================================

# Code for preparing the cancer clinical  reports for analysis by LLMs

# =============================================================================
 
# =============================================================================
# Imports
# ============================================================================= 
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from collections import Counter
import csv

import sys
sys.path.append('../../../')

# Internal Files
from LLMCore.APIs.Utils.Utils import Utils
# =============================================================================

# =============================================================================
# Class containing functions to preprocess Cancer Clinical Reports Dataset
# =============================================================================
class CCR_DataPrep():
    
    
    #--------------------------------------------------------------------------        
    # Init
    #--------------------------------------------------------------------------        
    def __init__(self):
        self.UtilObj = Utils()

    #--------------------------------------------------------------------------       
    
    #--------------------------------------------------------------------------        
    # Split data
    #-------------------------------------------------------------------------- 
    def split_data(self, data, train_val_per = 0.8, seed = 9):
       # Split data into training-validation and test set
       # Check if save path exists using Utils folder check
        
        rng = np.random.default_rng(seed = seed)
        
        #-- Data split sizes --#
        total_size = np.size(data,0)
        train_size = int(total_size*train_val_per)
        test_size = int(total_size-train_size)
        
        
        #-- Get split data --#
        index_arr = np.arange(0,total_size)
        train_val_chk = rng.choice(index_arr, size = train_size, replace = False, axis = 0)
        test_chk = np.array([x for x in index_arr if x not in train_val_chk])
        rng.shuffle(test_chk)
        
        
        raw_train_val_df = data.loc[train_val_chk,:]
        raw_test_df = data.loc[test_chk,:]
        
        raw_train_val_df = raw_train_val_df.reset_index()
        raw_test_df =  raw_test_df.reset_index()
        
        return(raw_train_val_df, raw_test_df)
    #--------------------------------------------------------------------------     
    
    #--------------------------------------------------------------------------        
    # Save all unique headers from the synoptic reports for clean up
    #--------------------------------------------------------------------------     
    def get_uniq_headers(self, data, data_path = ''):
        #--Prep data for manual cleanup--#
        
        keys_arr = []
        for ii in range(0,np.size(data,0),1):
            
            syn_rep = data.loc[ii,'SYNOPTIC_REPORT']
            if(isinstance(syn_rep, dict)):
                keys_arr.append(list(syn_rep.keys()))
        
        keys_arr_1d = [item for sublist in keys_arr for item in sublist]
        
        
        unique_keys = Counter(keys_arr_1d)
        
        unique_keys_dict = dict(unique_keys)
        
        if(data_path != ''):
            with open(data_path, 'w') as f:
                w = csv.writer(f)
                w.writerows(unique_keys_dict.items())
            
        return(unique_keys_dict)
    #--------------------------------------------------------------------------     
    
    #--------------------------------------------------------------------------        
    # Clean up headers of synoptic reports in the data
    #--------------------------------------------------------------------------     
    def get_clean_headers(self, data, clean_dict):    
        
        for ii in range(0,np.size(data,0),1):
        
            syn_rep = data.loc[ii,'SYNOPTIC_REPORT']
            
            if(isinstance(syn_rep, dict)):
                new_syn_rep = {}
                for key in syn_rep.keys():
                    
                    if(key in clean_dict):
                        
                        if(pd.isnull(clean_dict[key])):
                            new_key = str(key).upper()
                        else:
                            new_key = clean_dict[key]
                        
                    else:
                        new_key = key.upper()
                    
                    new_syn_rep[new_key] = syn_rep[key]
                    
                    
                data.at[ii,'SYNOPTIC_REPORT'] = new_syn_rep        
    
    
        return(data)
    #--------------------------------------------------------------------------        
    
    
    
    #--------------------------------------------------------------------------        
    # Prep data for manual clean up by storing in a csv format
    #--------------------------------------------------------------------------     
    def prep_manual_cleanup(self):    
        
    
    
        return()
    #--------------------------------------------------------------------------     
        
    
    #--------------------------------------------------------------------------        
    # Assemble questions
    #--------------------------------------------------------------------------        
    def get_question_df(self):
        question_df = pd.DataFrame([])
        
        question_df['question'] = []
        question_df['dict_ref'] = []
        
        ii = 0
        
        
        #---- Classification ----#        
        question_df.loc[ii,'question'] = 'What is the distant metastasis?'
        question_df.loc[ii,'dict_ref'] = 'DISTANT METASTASIS'
        ii = ii + 1       
        
        
        question_df.loc[ii,'question'] = 'What is the laterality?'
        question_df.loc[ii,'dict_ref'] = 'LATERALITY'
        ii = ii + 1         
        
        
        question_df.loc[ii,'question'] = 'What is the lymphovascular invasion?'
        question_df.loc[ii,'dict_ref'] = 'LYMPHOVASCULAR INVASION'
        ii = ii + 1     
                

        question_df.loc[ii,'question'] = 'What are the pathologic staging descriptors?'
        question_df.loc[ii,'dict_ref'] = 'PATHOLOGIC STAGING DESCRIPTORS'
        ii = ii + 1 
        
        
        question_df.loc[ii,'question'] = 'What is the perineural invasion?'
        question_df.loc[ii,'dict_ref'] = 'PERINEURAL INVASION'
        ii = ii + 1 
        
        
        question_df.loc[ii,'question'] = 'What is the primary tumor?'
        question_df.loc[ii,'dict_ref'] = 'PRIMARY TUMOR'
        ii = ii + 1          
        
        
        question_df.loc[ii,'question'] = 'What are the regional lymph nodes?'
        question_df.loc[ii,'dict_ref'] = 'REGIONAL LYMPH NODES'
        ii = ii + 1         
        
        
        question_df.loc[ii,'question'] = 'What is the speciment integrity?'
        question_df.loc[ii,'dict_ref'] = 'SPECIMEN INTEGRITY'
        ii = ii + 1            
        #-------------------------------------------------------------------#
        
        
                
        #---- QA ----#
        
        # question_df.loc[ii,'question'] = 'What is the lymph node sampling?'
        # question_df.loc[ii,'dict_ref'] = 'LYMPH NODE SAMPLING'
        # ii = ii + 1                   
        
        # question_df.loc[ii,'question'] = 'What is the total number examined?'
        # question_df.loc[ii,'dict_ref'] = 'NUMBER EXAMINED'
        # ii = ii + 1         
                
        # question_df.loc[ii,'question'] = 'What is the total number involved?'
        # question_df.loc[ii,'dict_ref'] = 'NUMBER INVOLVED'
        # ii = ii + 1         
        
        # question_df.loc[ii,'question'] = 'What are the surgical margins?'
        # question_df.loc[ii,'dict_ref'] = 'SURGICAL MARGINS'
        # ii = ii + 1            
                
        
        # question_df.loc[ii,'question'] = 'What is the tumor focality?'
        # question_df.loc[ii,'dict_ref'] = 'TUMOR FOCALITY'
        # ii = ii + 1   

        # question_df.loc[ii,'question'] = 'What is the tumor site?'
        # question_df.loc[ii,'dict_ref'] = 'TUMOR SITE'
        # ii = ii + 1          
        
        # question_df.loc[ii,'question'] = 'What is the tumor size?'
        # question_df.loc[ii,'dict_ref'] = 'TUMOR SIZE'
        # ii = ii + 1 

        #-------------------------------------------------------------------#


        #---- Generation ----#
        # question_df.loc[ii,'question'] = 'What is the histologic grade?'
        # question_df.loc[ii,'dict_ref'] = 'HISTOLOGIC GRADE'
        # ii = ii + 1         
       
        
        # question_df.loc[ii,'question'] = 'What is the histologic type?'
        # question_df.loc[ii,'dict_ref'] = 'HISTOLOGIC TYPE'
        # ii = ii + 1   
        
        
        # question_df.loc[ii,'question'] = 'What is the mitotic rate?'
        # question_df.loc[ii,'dict_ref'] = 'MITOTIC RATE'
        # ii = ii + 1

              
        # question_df.loc[ii,'question'] = 'What is the procedure?'
        # question_df.loc[ii,'dict_ref'] = 'PROCEDURE'
        # ii = ii + 1 
                        
        
        # question_df.loc[ii,'question'] = 'What is the protocol biopsy?'
        # question_df.loc[ii,'dict_ref'] = 'PROTOCOL BIOPSY'
        # ii = ii + 1          
                    

        # question_df.loc[ii,'question'] = 'What is the specimen?'
        # question_df.loc[ii,'dict_ref'] = 'SPECIMEN'
        # ii = ii + 1 
        
        
        # question_df.loc[ii,'question'] = 'What is the treatment effect?'
        # question_df.loc[ii,'dict_ref'] = 'TREATMENT EFFECT'
        # ii = ii + 1 
                
        #--------------------------------------------------------------------#
        #--- Unclear if needed --#
        
        
        # question_df.loc[ii,'question'] = 'What is the pathologic staging?'
        # question_df.loc[ii,'dict_ref'] = 'PATHOLOGIC STAGING (AJCC, 7TH EDITION)'
        # ii = ii + 1     
                    
        
        # question_df.loc[ii,'question'] = 'What is the microscopic extent of tumor?'
        # question_df.loc[ii,'dict_ref'] = 'MICROSCOPIC EXTENT OF TUMOR'
        # ii = ii + 1        
        
                
        # question_df.loc[ii,'question'] = 'What is the total?'
        # question_df.loc[ii,'dict_ref'] = 'TOTAL'
        # ii = ii + 1              

        
        # question_df.loc[ii,'question'] = 'What are the margins?'
        # question_df.loc[ii,'dict_ref'] = 'MARGINS'
        # ii = ii + 1            
        

        #------
    
        # Less than 1000 samples available:
            
        # question_df.loc[ii,'question'] = 'What is the specimen size?'
        # question_df.loc[ii,'dict_ref'] = 'SPECIMEN SIZE'
        # ##question_df.loc[ii,'dict_ref'] = 'Specimen Size'
        # ii = ii + 1     
        

        # question_df.loc[ii,'question'] = 'What is the maximum tumor depth?'
        # question_df.loc[ii,'dict_ref'] = 'MAXIMUM TUMOR DEPTH'
        # ##question_df.loc[ii,'dict_ref'] = 'Maximum Tumor Depth'
        # ii = ii + 1 
        
        
        
        # question_df.loc[ii,'question'] = 'What is the tumor description?'
        # question_df.loc[ii,'dict_ref'] = 'TUMOR DESCRIPTION'
        # ##question_df.loc[ii,'dict_ref'] = 'Tumor Description (Gross Subtype)'
        # ii = ii + 1 
        

        # question_df.loc[ii,'question'] = 'What is the lymph node extracapsular extension?'
        # question_df.loc[ii,'dict_ref'] = 'LYMPH NODE EXTRACAPSULAR EXTENSION'
        # ##question_df.loc[ii,'dict_ref'] = 'Lymph Node Extracapsular Extension'
        # ii = ii + 1 
        
        # question_df.loc[ii,'question'] = 'What is the extracapsular extension?'
        # question_df.loc[ii,'dict_ref'] = 'EXTRACAPSULAR EXTENSION'
        # ##question_df.loc[ii,'dict_ref'] = 'Extracapsular extension'
        # ii = ii + 1 
        
        # question_df.loc[ii,'question'] = 'What is the size of the largest metastatic focus?'
        # question_df.loc[ii,'dict_ref'] = 'SIZE OF LARGEST METASTATIC FOCUS'
        # ##question_df.loc[ii,'dict_ref'] = 'Size of largest metastatic focus'
        # ii = ii + 1 
        

        
        return(question_df)
    #--------------------------------------------------------------------------          
    

        
    
    
    #--------------------------------------------------------------------------        
    # Code for preparing synoptic reports for QA fine tuning with segmented data
    #--------------------------------------------------------------------------   
    def convert_to_qa_df(self, raw_data_df, question_df, max_input_size):
    
        raw_data_df = raw_data_df.reset_index(drop=True)    
        
        qa_data_df = pd.DataFrame([])
        
        qa_data_df['doc_id'] = []
        qa_data_df['question'] = []
        qa_data_df['context'] = []
        
        qa_data_df['answers'] = []
        qa_data_df['answers'] = qa_data_df['answers'].astype('object')
        qa_data_df['nlp_raw_ref'] = []
        qa_data_df['nlp_raw_ref'] = qa_data_df['nlp_raw_ref'].astype('object')        
        
        N_samples = np.size(raw_data_df,0)
        
        ii = 0
        jj = 0
        lcl_cnt = 0
        for jj in tqdm(range(0,np.size(question_df,0),1), desc = "outer loop", leave = False):
            question = question_df.loc[jj,'question']
                
            for ii in tqdm(range(0,N_samples,1), desc = "inner loop"):
                    
                full_context = str(raw_data_df.loc[ii,'DIAGNOSIS']) \
                    + str(raw_data_df.loc[ii,'GROSS_DESCRIPTION']) \
                        + str(raw_data_df.loc[ii,'FROZEN_SECTION'])
                        
                clean_full_context = full_context.replace('\r\n  ', '')
                
                
                syn_report = raw_data_df.loc[ii,'SYNOPTIC_REPORT']            
                try:              
                    
                    no_segments = int(np.ceil(len(clean_full_context)/max_input_size))
                    
                    seg_start = 0
                    seg_end = 0
                    for kk in range(0,no_segments,1):
                        
                        
                        
                        if(kk==no_segments-1):            
                            seg_end = seg_start + (len(clean_full_context)%max_input_size)
                        
                        else:
                            seg_end = seg_start + max_input_size                   
                        
                        if(seg_end > seg_start):
                            context_segment = clean_full_context[seg_start:seg_end]
                            
                            answer = syn_report[question_df.loc[jj,'dict_ref']]
                            answer = answer[0:-1]
                            
                            answer_st_loc = context_segment.find(answer)
                            
                            answer_dict = {}
                            if(answer_st_loc == -1):    
                                answer_dict['text'] = ['Answer not in text']
                            else:
                                answer_dict['text'] = [answer]
                            answer_dict['answer_start'] = [answer_st_loc]
                            
                            qa_data_df.loc[lcl_cnt,'doc_id'] = raw_data_df.loc[ii,'DOC_ID']
                            qa_data_df.loc[lcl_cnt,'question'] = question
                            qa_data_df.loc[lcl_cnt,'context'] = context_segment
                            qa_data_df.at[lcl_cnt, 'answers'] = answer_dict #[answer_dict]
                            qa_data_df.at[lcl_cnt, 'nlp_raw_ref'] = answer
                        
                            seg_start = seg_start + max_input_size
                            lcl_cnt = lcl_cnt + 1
                
                except:
                    # print('Error encountered, skipping file')
                    a = 1
    
        return(qa_data_df)
    #--------------------------------------------------------------------------       
        
    
    #--------------------------------------------------------------------------        
    # Preprocess data to tokenize the data (Should likely be called prepare 
    # training or finetuning data)
    #--------------------------------------------------------------------------       
    def tokenize_qa_data(self, examples, max_length = 500):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length= max_length,#384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []
    
        for i, offset in enumerate(offset_mapping):
            try:
                answer = answers[i]
                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)
        
                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1
        
                # If the answer is not fully inside the context, label it (0, 0)
                if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)
        
                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)
                    
            except:
                print('Error encountered, skipping file')                
    
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        
        return inputs    
    #-------------------------------------------------------------------------- 
    
    
    #--------------------------------------------------------------------------
    # Code for creating text files from a list of words for CancerBERT
    #-------------------------------------------------------------------------- 
    def create_EXT_cancerBERT_testfile(self, word_list, file_name, location):
        
        # Make labelled string for text file for CancerBERT NER input file format
        new_word_list = []
        for ii in range(0,len(word_list),1):
            
            word = word_list[ii]
            
            if(word.endswith('.')):
                word_new = word.replace('.','') + ' O'
                new_word_list.append(word_new)
                new_word_list.append('. O')
                new_word_list.append('')
                new_word_list.append('')
            elif(word == ''):
                new_word = word
            else:
                word_new = word + ' O'
                new_word_list.append(word_new)
            
        new_word_list.append('')
        new_word_list.append('')
        with open(location + file_name, 'w') as file:
            for word in new_word_list:
                file.write(word + '\n')
    
        return()
    #--------------------------------------------------------------------------    
    
    
    #--------------------------------------------------------------------------
    # Code for synoptic report data for analysis using External Tool CancerBERT 
    #--------------------------------------------------------------------------   
    def convert_to_EXT_CancerBERT_data(self, raw_data_df, data_elements_list):
        
        raw_data_df = raw_data_df.reset_index(drop=True)    
        
        
        qa_data_df = pd.DataFrame([])
        qa_data_df['doc_id'] = []
        qa_data_df['context'] = []
        for jj in range(0,len(data_elements_list),1):
            qa_data_df[data_elements_list[jj]] = []
        
        N_samples = np.size(raw_data_df,0)
        
        ii = 0
        for ii in tqdm(range(0,N_samples,1)):
            
            
            full_context = str(raw_data_df.loc[ii,'DIAGNOSIS']) \
                + str(raw_data_df.loc[ii,'GROSS_DESCRIPTION']) \
                    + str(raw_data_df.loc[ii,'FROZEN_SECTION'])
                    
            clean_full_context = full_context.replace('\r\n  ', '')
            syn_report = raw_data_df.loc[ii,'SYNOPTIC_REPORT']
            
            qa_data_df.loc[ii,'doc_id'] = raw_data_df.loc[ii,'DOC_ID']
            qa_data_df.loc[ii,'context'] = clean_full_context
            
            for jj in range(0,len(data_elements_list),1):
                
                try:
                    qa_data_df.loc[ii, data_elements_list[jj]] = syn_report[data_elements_list[jj]]
                except:
                    qa_data_df.loc[ii, data_elements_list[jj]] = -1
                    
    
        return(qa_data_df)
    #--------------------------------------------------------------------------    
    
    #--------------------------------------------------------------------------
    # Code for creating test files for analysis using External Tool CancerBERT 
    #--------------------------------------------------------------------------   
    def make_EXT_cancerBERT_testfiles(self, qa_data_df, max_seq_length, data_elements_list):
    
        ii = 0
        master_cnt = 0
        cbert_key_df = pd.DataFrame([])
        cbert_key_df['doc_id'] = []
        cbert_key_df['file_name'] = []
        
        for jj in range(0,len(data_elements_list),1):
            cbert_key_df[data_elements_list[jj]] = []
        
        location = "../../../../../Imp/Data/ProcessedDataCBERT/InputFiles/"
        
        
        for ii in tqdm(range(0,np.size(qa_data_df,0),1)):
          
            doc_id = qa_data_df.loc[ii,'doc_id']
            context = qa_data_df.loc[ii,'context']
            
            context_words = context.split(' ')
            
            no_files = int(np.ceil(len(context_words)/max_seq_length))
            
            seg_start = 0
            seg_end = 0
            lcl_cnt = 0
            for kk in range(0,no_files,1):
                    
                if(kk==no_files-1):            
                    seg_end = seg_start + (len(context_words)%max_seq_length)
                
                else:
                    seg_end = seg_start + max_seq_length                   
            
                
                context_segment = context_words[seg_start:seg_end]
                file_name = str(doc_id) + '_' + str(lcl_cnt) + '.txt'
                self.create_EXT_cancerBERT_testfile(context_segment, file_name, location)
            
                seg_start = seg_start + max_seq_length
                lcl_cnt = lcl_cnt + 1
            
            
                cbert_key_df.loc[master_cnt, 'doc_id'] = doc_id
                cbert_key_df.loc[master_cnt, 'file_name'] = file_name
                for jj in range(0,len(data_elements_list),1):
                    cbert_key_df.loc[master_cnt, data_elements_list[jj]] = qa_data_df.loc[ii, data_elements_list[jj]]
                
                
                master_cnt = master_cnt + 1
    
        return(cbert_key_df)
    #--------------------------------------------------------------------------    
    
    
    #--------------------------------------------------------------------------        
    # 
    #--------------------------------------------------------------------------   

    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------        
    #
    #--------------------------------------------------------------------------   



    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------        
    #
    #--------------------------------------------------------------------------   



    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------        
    #
    #--------------------------------------------------------------------------   



    #--------------------------------------------------------------------------    
    
# =============================================================================
