import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import  AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import argparse

from openprompt import PromptDataLoader
from openprompt.plms import load_plm
from openprompt.data_utils import InputExample
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt import PromptForClassification

from verbalizer import ManualVerbalizer, KnowledgeableVerbalizer

import json
from transformers.tokenization_utils import PreTrainedTokenizer
from yacs.config import CfgNode
from openprompt.data_utils import InputFeatures
import re
from openprompt import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from openprompt.utils.logging import logger
import os
from openprompt.prompts.manual_template import ManualTemplate
from transformers.utils.dummy_pt_objects import PreTrainedModel



import os

parser = argparse.ArgumentParser(description='')
args   = parser.parse_args()
args.device         = torch.device('cuda:1') 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

args.BASE = 'roberta-large'
args.templates = 'Adapted_t5_large_Friends_'
args.verbalizer = 'posterior_'
MAX_LEN = 128
use_cuda = True







def test(input_sample, prompt_model, mytemplate, tokenizer, WrapperClass):
    prompt_model.eval()
    labels_list = np.array([])
    pred_list = np.array([])
    overall_logits = []
    
    
    one_sample_loader = PromptDataLoader(
        dataset                 = [input_sample], 
        template                = mytemplate, 
        tokenizer               = tokenizer,
        tokenizer_wrapper_class = WrapperClass, 
        max_seq_length          = MAX_LEN, 
        batch_size              = 1, 
        shuffle                 = True, 
        teacher_forcing         = False, 
        predict_eos_token       = False,
        truncate_method         = "head")
    for step, inputs in enumerate(one_sample_loader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
#     print(logits)
    return logits


personality_dict = {
    'A': 'Agreeableness',
    'C': 'Conscientiousness',
    'E': 'Extraversion',
    'O': 'Openness',
    'N': 'Neuroticism'
}


if __name__ == '__main__':

    input_sent = 'I am happy.'
    input_sample = InputExample(text_a=input_sent, label=0,guid=0)
    
    ans_list = []


    personalities = ['A', 'C', 'E', 'O', 'N']

    for personality in personalities:
        args.personality = personality

        # ***** construct verbalizer ***** 
        n = 200
        with open('label_words/'+args.verbalizer+args.personality+'_words.txt', 'r') as f_verbalizer:
                pos = [i.strip() for i in f_verbalizer.readline().split(',')][:n]
                neg = [i.strip() for i in f_verbalizer.readline().split(',')][:n]

        with open('label_words/'+args.verbalizer+args.personality+'_weights.txt', 'r') as f_verbalizer:
                pos_weights = eval(f_verbalizer.readline())[:n]
                neg_weights = eval(f_verbalizer.readline())[:n]

        diff_len = len(neg_weights) - len(pos_weights)

        if diff_len >= 0:
            label_words_weights = torch.Tensor([neg_weights, pos_weights + [0]*diff_len])
        else:
            label_words_weights = torch.Tensor([neg_weights + [0]*(-diff_len), pos_weights])



        candidate_templates = []
        with open('templates/'+args.templates+args.personality+'_templates_top_10.txt', 'r') as f_template:
            candidate_templates = [i.strip() for i in f_template.readlines()]

        logits_all_templates = []

        plm, tokenizer, model_config, WrapperClass = load_plm("roberta", args.BASE)
        wrapped_tokenizer = WrapperClass(max_seq_length=MAX_LEN, tokenizer=tokenizer,truncate_method="head")

        for template in candidate_templates:

            mytemplate = ManualTemplate(
                    text = template,
                    tokenizer = tokenizer,
            )

            class_labels = [0,1]

            myverbalizer = ManualVerbalizer(
                classes = class_labels,
                label_words = {
                    0 : neg, 
                    1 : pos
                },
                tokenizer=tokenizer,
                label_words_weights=label_words_weights.cuda())

            prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
            prompt_model = prompt_model.cuda()
            
            
            logits_ = test(input_sample, prompt_model, mytemplate, tokenizer, WrapperClass)

        logits_all_templates.append(logits_.unsqueeze(dim=0))
        logits_all_templates = torch.cat(logits_all_templates)
        logits_all_templates = torch.mean(logits_all_templates, 0).squeeze(0)
        neg = logits_all_templates.cpu()[0]
        pos = logits_all_templates.cpu()[1]
        
        if pos >= neg: 
            ans_list.append(personality_dict[args.personality])
        else:
            ans_list.append('Not '+ personality_dict[args.personality])
    

    print('The input sentence is: ' + input_sent)
    print('The big five personalities of the speaker is:', ans_list[0], ', ',  ans_list[1], ', ', ans_list[2], ', ', ans_list[3], ', ', ans_list[4], ', ', 'respectively.')

        











































