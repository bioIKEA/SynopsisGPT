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
import random 

import tensorflow as tf

# import datasets
import jellyfish

from transformers import pipeline, AutoModel
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import TFAutoModelForQuestionAnswering
from transformers import create_optimizer


import sys
sys.path.append('../')


#--Llama 2 finetune imports --#
import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
# from datasets import load_dataset
#-----------------------------#


#-- Internal Imports --#
from APIs.Utils.Utils import Utils
from APIs.DataPrep.CCR_DataPrep import CCR_DataPrep
from APIs.Metrics.LLM_Metrics import LLM_Metrics
#---------------------------------------------------#

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

#-- Code for finetuning llama models --#
#-- Reproducibility --#
seed = 42
set_seed(seed)
#---------------------#



#--Instantiate class objects--#
UtilObj = Utils()
DPObj = CCR_DataPrep()
METObj = LLM_Metrics()

#%%

reprocess_data = 0
get_new_qdf = 0

#-- Separate data quesiton, context, answers --#
max_input_size = 3795

# data_sel = ''
# data_sel = '_cl'
data_sel = '_qa'
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
    raw_train_val_df = np.load(load_path + load_name + '_train_val_df.pkl', allow_pickle = True)
    #-- Test data --#
    raw_test_df = np.load(load_path + load_name +'_test1_df.pkl', allow_pickle = True)
    
    qa_data_train_val_df = DPObj.convert_to_qa_df(raw_train_val_df, question_df, max_input_size)
    qa_data_test_df = DPObj.convert_to_qa_df(raw_test_df, question_df, max_input_size)

else:
    # For loading previously prepared data
    load_path = "DataPrep/"
    load_name = "qa_data_size_"+str(max_input_size)+ str(data_sel)
    
    
    qa_data_train_val_df = np.load(load_path + load_name + '_train_val_df.pkl', allow_pickle = True)
    qa_data_test_df = np.load(load_path + load_name + '_test1_df.pkl', allow_pickle = True)

#-- End of special operations for cancer data --#


#%%
# Converting data to a format that works
import pyarrow as pa
# import pyarrow.dataset as ds
# import pandas as pd
from datasets import Dataset
from sklearn.utils import shuffle

# Training dataset
N_train = -1 # If set to -1, selects all the data
if(N_train == -1):
    dat_train = qa_data_train_val_df
else:
    dat_train = qa_data_train_val_df.groupby('question').apply(lambda x:x.sample(N_train)).reset_index(drop = True)
    dat_train = pd.DataFrame(dat_train)

N_steps = N_train

mdl_sp_tag = 'SPQA_v2'

## qa_data_train_val_df = shuffle(qa_data_train_val_df).reset_index(drop = True)
## dataset_train = Dataset(pa.Table.from_pandas(qa_data_train_val_df))

dataset_train = Dataset(pa.Table.from_pandas(dat_train))
dataset_test = Dataset(pa.Table.from_pandas(qa_data_test_df))


#%%
# Llama 2 model
import transformers
import torch


model_str = "meta-llama/Llama-2-7b-hf"
# model_str = "meta-llama/Llama-2-13b-hf"
# model_str = "meta-llama/Llama-2-70b-hf"
# 
model_save_name = model_str.replace('/','_') + '_FT_'+str(max_input_size)+'tokens_'+ str(data_sel)+str(N_steps)+'steps_'+str(mdl_sp_tag)

#%% From https://github.com/ovh/ai-training-examples/blob/main/notebooks/natural-language-processing/llm/miniconda/llama2-fine-tuning/llama_2_finetuning.ipynb


# Downloads model and tokenizer -KEEP
def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{40960}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer



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
    
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['question']}"
    input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
    response = f"{RESPONSE_KEY}\n{sample['nlp_raw_ref']}" if sample["nlp_raw_ref"] else None
    end = f"{END_KEY}"
    
    
    
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    formatted_prompt = "\n\n".join(parts)
    
    sample["text"] = formatted_prompt
        
    # except:
    #     print('error')

    return sample


# print(create_prompt_formats(sample)["text"])
#%%

# Finds the max length of the model - KEEP
# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


# Tokenize function - eval code uses pipeline, so we will use this now
# Processed prompt data should be stored as a df or dict with "text" as keyword - KEEP
def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# Tokenizes the dataset - KEEP
# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)#, batched=True)
    
    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        # remove_columns=["instruction", "context", "response", "text", "category"],
        # remove_columns= ['doc_id', 'question', 'context', 'answers', 'nlp_raw_ref','__index_level_0__', 'text'],#qa_data_test_df.columns,
        remove_columns= ['doc_id', 'question', 'context', 'answers', 'nlp_raw_ref', 'text'],#qa_data_test_df.columns,

        # remove_columns = pd.DataFrame(dataset).columns,
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset




#%%
# Creates function to use LLM in 4bit version - KEEP
def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config



# Creates parameter efficient fine tuning configuration - KEEP
def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config

# Further set-up for efficient training - KEEP
# SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


# Further diagnosis for printing trainable parameters - KEEP
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    
#%%
# Loads the appropriate llama model - KEEP
# Load model from HF with user's token and with bitsandbytes config
model_name = model_str #"meta-llama/Llama-2-7b-hf" 
# model_name = "meta-llama/Llama-2-13b-hf"
# model_name = "meta-llama/Llama-2-70b-hf"

bnb_config = create_bnb_config()
model, tokenizer = load_model(model_name, bnb_config)


## Preprocess dataset
max_length = get_max_length(model)
dataset_train = preprocess_dataset(tokenizer, max_length, seed, dataset_train)



#%%
# N_steps = 64#np.size(dat_train,0)
num_epochs = 1
# Train function - KEEP
def train(model, tokenizer, dataset, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)
    
    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)
    
    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            # max_steps=N_steps,
            do_eval = True, 
            eval_steps = 300,
            num_train_epochs = num_epochs,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=300,
            output_dir="outputs2",
            optim="paged_adamw_8bit",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs
    
    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training
    
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)
     
    do_train = True
    
    # Launch trainingl
    print("Training...")
    
    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)    
    
    ###
    
    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    
    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()
    
    
output_dir = "TrainedModels/llama2/final_checkpoint2"

train(model, tokenizer, dataset_train, output_dir)

#%%

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = "TrainedModels/llama2/final_merged_checkpoint2"
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)

# save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_merged_dir)

#%%

output_merged_dir = "TrainedModels/llama2/Fine_tuned/"
os.makedirs(output_merged_dir + model_save_name, exist_ok=True)
model.save_pretrained(output_merged_dir +  model_save_name, safe_serialization=True)

# save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_merged_dir + model_save_name)


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

dataset_test_df = dataset_test.map(create_test_prompt_formats)
dataset_test_df = pd.DataFrame(dataset_test_df)



#%%

# Inference - MOVE to different file later

# model_name_clean = model_str.replace('/','_')

# Clear cuda cache
# torch.cuda.empty_cache()

# Specify device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#-- Evaluate results on the test data --#
preds = pd.DataFrame([])

preds['DOC_ID'] = []
preds['QUESTION'] = []
preds['CONTEXT'] = []
preds['NLP_RAW_REFERENCE'] = []
preds['NLP_REFERENCE'] = []
# preds[model_str] = []
preds['MODEL_OUT'] = []
preds['MANUAL_CHK'] = []




raw_test_df = dataset_test_df.reset_index(drop = True)
lcl_cnt = 0 

N_test_files = 100 #Number of unique files to test

for jj in range(0,np.size(question_df,0),1):
    question = question_df.loc[jj,'question']
    max_new_tokens = 30#question_df.loc[jj,'max_new_tokens'] 

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
    
    
    
    
    # Sample prompt for 1 shot QA
    # sample_context =  sub_n_raw_test_df.loc[0,'context'].replace('\r\n  ','')
    # sample_answer = sub_n_raw_test_df.loc[0,'answers']['text'][0]        
    # sample_text = 'Given this context: \n"' + sample_context +'"\n  Answer the following question in less than 3 words: "' + question + '"\n  Answer:  ' + sample_answer + '\n\n'    
             
    
    
    for ii in tqdm(range(0,N_sub_elements,1)):
                
        try:
            # context = sub_n_raw_test_df.loc[ii,'context'].replace('\r\n  ','')
            context = sub_n_raw_test_df.loc[ii,'text']
            syn_report = sub_n_raw_test_df.loc[ii,'answers']['text']               
            
            #-- Predict the answer --#
    
            context_len = len(context)
            
    
        
            # Tokenize input text
            inputs = tokenizer(context, return_tensors="pt").to(device)
            
            # Get answer
            # (Adjust max_new_tokens variable as you wish (maximum number of tokens the model can generate to answer the input))
            outputs = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"], 
                                     max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
            
            
            # Decode output & print it
            gen_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        
        
            # Save answers
            preds.loc[lcl_cnt,'DOC_ID'] = sub_n_raw_test_df.loc[ii,'doc_id']
            preds.loc[lcl_cnt,'QUESTION'] = question
            preds.loc[lcl_cnt,'CONTEXT'] = context
            preds.loc[lcl_cnt,'NLP_RAW_REFERENCE'] = sub_n_raw_test_df.loc[ii,'nlp_raw_ref']
            preds.loc[lcl_cnt, 'NLP_REFERENCE'] = syn_report[0]#syn_report[question_df.loc[jj,'dict_ref']]
            preds.loc[lcl_cnt, 'MODEL_OUT'] = str(gen_sequence[context_len::])#tokenizer.decode(predict_answer_tokens)
            preds.loc[lcl_cnt, 'MANUAL_CHK'] = 0
            preds.loc[lcl_cnt,'JARO_SIMILARITY'] = jellyfish.jaro_similarity(preds.loc[lcl_cnt, 'MODEL_OUT'], preds.loc[lcl_cnt,'NLP_RAW_REFERENCE'])
            preds.loc[lcl_cnt,'JACCARD_SIMILARITY'] = METObj.jaccard_similarity(preds.loc[lcl_cnt, 'MODEL_OUT'], preds.loc[lcl_cnt,'NLP_RAW_REFERENCE'])
                
            
            
            lcl_cnt = lcl_cnt + 1
        
        except:
            print('Error encountered, skipping file')

#%%
# model_save_name = model_name.replace('/','_') + '_FT_'+str(max_input_size)+'tokens_'+str(N_steps)+'steps_'+'v4_Test'

preds.to_pickle('Results/'+model_save_name+'_'+str(N_test_files)+'_preds'+'.pkl')
question_df.to_pickle('Results/'+model_save_name+'_'+str(N_test_files)+'_qdf'+'.pkl')
preds.to_csv(path_or_buf = 'Results/'+model_save_name+'_'+str(N_test_files)+'_PREDS.csv')


    


#%%
'''
path_to_model = 'TrainedModels/llama2/Fine_tuned/meta-llama_Llama-2-7b-hf_FT_3775tokens_5steps_v4_Test/'


from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained(path_to_model)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(path_to_model)

# Example text
text = "Your input text here"

# Tokenize and generate input IDs
inputs = tokenizer(text, return_tensors="pt")

# Generate prediction
output = model.generate(**inputs, max_new_tokens = 10)

# Decode output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output)

'''