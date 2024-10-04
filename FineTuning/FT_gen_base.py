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
# from transformers import TFAutoModelForQuestionAnswering
from transformers import create_optimizer
# from transformers import DataCollatorForLanguageModeling
# from transformers import AutoModelForMaskedLM
from transformers import AutoModelForCausalLM

import sys
sys.path.append('../')

#-- Internal Imports --#
from APIs.Utils.Utils import Utils
from APIs.DataPrep.CCR_DataPrep import CCR_DataPrep
from APIs.Metrics.LLM_Metrics import LLM_Metrics
#---------------------------------------------------#

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

data_sel = ''
# data_sel = '_cl'
# data_sel = '_rcl'
# data_sel = '_qa'
# data_sel = '_gen'


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
# Converting data to a format that works
import pyarrow as pa
from datasets import Dataset
from sklearn.utils import shuffle

# Training dataset
N_train = 100 # If set to -1, selects all the data
if(N_train == -1):
    dat_train = qa_data_train_val_df
else:
    dat_train = qa_data_train_val_df.groupby('question').apply(lambda x:x.sample(N_train)).reset_index(drop = True)
    dat_train = pd.DataFrame(dat_train)
    
N_steps = N_train


dataset_train = Dataset(pa.Table.from_pandas(dat_train))
dataset_test = Dataset(pa.Table.from_pandas(qa_data_test_df))

#%%


from transformers import AutoModelForCausalLM, AutoTokenizer

# model_str =  "distilbert-base-cased-distilled-squad" 
# model_str = "EleutherAI/gpt-j-6B"
# model_str = "facebook/bart-base"
# model_str = "facebook/bart-large-cnn"
# model_str  = "gpt2"
model_str = "bert-large-uncased"
# model_str = "bert-base-uncased"
model_name = model_str +'_FT_'+str(max_input_size)+'tokens_'+\
    str(data_sel)+str(N_steps)+ '_FT_MAINTEST_v1'

# model_checkpoint = "distilbert-base-cased-distilled-squad"  # Or any other model
model = AutoModelForCausalLM.from_pretrained(model_str)
tokenizer = AutoTokenizer.from_pretrained(model_str)

tokenizer.pad_token = tokenizer.eos_token

#%%

# Modified function for qa cancer clinical notes dataset
def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters 
    :param sample: Sample dictionnary
    """
    # try:
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "### Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    
    # blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['question']}"
    input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
    response = f"{RESPONSE_KEY}\n{sample['nlp_raw_ref']}" if sample["nlp_raw_ref"] else None
    # end = f"{END_KEY}"
    
    
    
    parts = [part for part in [input_context, instruction, response] if part]

    formatted_prompt = "\n\n".join(parts)
    
    sample["text"] = formatted_prompt
        
    # except:
    #     print('error')

    return sample


dataset_train = dataset_train.map(create_prompt_formats)

#%%


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# dataset["train"] = dataset["validation"]

tokenized_datasets = dataset_train.map(tokenize_function, batched=True)


from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm= False)

#%%
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",           # Output directory
    num_train_epochs=1,               # Total number of training epochs
    per_device_train_batch_size=2,   # Batch size per device during training
    per_device_eval_batch_size=2,    # Batch size for evaluation
    warmup_steps=2,                 # Number of warmup steps for learning rate scheduler
    # weight_decay=0.01,                # Strength of weight decay
    # logging_dir='./logs',             # Directory for storing logs
    learning_rate=2e-4,
    logging_steps=10,
)



from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer = tokenizer,
    train_dataset=tokenized_datasets,#tokenized_datasets["train"],
    eval_dataset=tokenized_datasets,#["validation"],
)


print('Training...')
train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
print(metrics) 

# trainer.evaluate()
model.save_pretrained("TrainedModels/CLM/" +str(model_name))









