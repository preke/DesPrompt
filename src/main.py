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

from verbalizer import MyVerbalizer#, ManualVerbalizer


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


parser = argparse.ArgumentParser(description='')
args   = parser.parse_args()
import os
args.device         = torch.device('cuda:0') 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
args.adam_epsilon   = 1e-8
args.num_of_epoches = 3
args.num_class      = 2
args.drop_out       = 0.1
args.test_size      = 0.1
args.method         = 'Ours' #'PET_wiki', 'KPT_augmented'
args.learning_rate  = 2e-5
# knowledgeable prompt tuning:

args.candidate_frac  = 0.5
args.pred_temp       = 1.0
args.max_token_split = -1



# datasets      = ['MyPersonality', 'Pan', 'mbti', 'Friends_Persona', 'Essay']
datasets      = ['Friends_Persona']#[ 'Essay', 'MyPersonality', 'Pan'] # 'Friends_Persona',
seeds         = [321, 42, 1024, 0, 1, 13, 41, 123, 456, 999]
few_shot_list = [0]#, 1, 8, 16, 32, -1]

args.BASE = 'RoBERTa-large'
args.templates = 'Adapted_t5_large_Friends_' #'t5-large_Friends_' 
args.verbalizer = '' # 'posterior_'


MAX_LEN_dict = {
    'Friends_Persona': 128,
    'Essay'          : 512,
    'MyPersonality'  : 512,
    'Pan'            : 512,
    # 'mbti'           : 512
}

Personalities_dict = {
    'Friends_Persona': ['A', 'C', 'E', 'O', 'N'],
    'Essay'          : ['A', 'C', 'E', 'O', 'N'],
    'MyPersonality'  : ['A', 'C', 'E', 'O', 'N'],
    'Pan'            : ['A', 'C', 'E', 'O', 'N'],
    # 'mbti'           : ['I', 'N', 'T', 'J']
}


DATA_PATH_dict = {
    'Friends_Persona': '../data/FriendsPersona/Friends_',
    'Essay'          : '../data/Essay/Essay_',
    'MyPersonality'  : '../data/myPersonality/MyPersonality_',
    'Pan'            : '../data/pan2015/Pan_',
    # 'mbti'           : '../data/Kaggle_mbti/mbti_'

}
 


use_cuda = True
# ==============

def evaluation(validation_dataloader, prompt_model):
    prompt_model.eval()
    labels_list = np.array([])
    pred_list = np.array([])
    for step, inputs in enumerate(validation_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        preds  = torch.argmax(logits, dim=-1).cpu().tolist()
        labels_list = np.append(labels.cpu().tolist(), labels_list)
        pred_list = np.append(preds, pred_list)
    return f1_score(labels_list, pred_list)


def get_test_results(test_dataloader, logits):
    prompt_model.eval()
    labels_list = np.array([])
    pred_list = np.array(torch.argmax(logits, dim=-1).cpu().tolist())
    for step, inputs in enumerate(test_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        labels = inputs['label']
        labels_list = np.append(labels.cpu().tolist(), labels_list)
    print(pred_list)
    print('****************')
    print(labels_list)
    import time
    time.sleep(100)
    return f1_score(labels_list, pred_list)


def test(test_dataloader, prompt_model):
    prompt_model.eval()
    labels_list = np.array([])
    pred_list = np.array([])
    overall_logits = []
    for step, inputs in enumerate(test_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        overall_logits.append(logits.detach())
    overall_logits = torch.cat(overall_logits)
    # print('='*20)
    # print(overall_logits.shape)
    return overall_logits


def fit(model, train_dataloader, val_dataloader, loss_func, optimizer):
    best_score = 0.0
    for epoch in range(3):
        train_epoch(model, train_dataloader, loss_func, optimizer)
        score = evaluate(model, val_dataloader)
        if score > best_score:
            best_score = score
    return best_score



def data_loader(args, dataset, mytemplate, tokenizer, WrapperClass):
    
    if args.shots > 0:
        args.batch_size = 1
    else:
        args.batch_size = 1

    train_dataloader = PromptDataLoader(
        dataset                 = dataset["train"], 
        template                = mytemplate, 
        tokenizer               = tokenizer,
        tokenizer_wrapper_class = WrapperClass, 
        max_seq_length          = MAX_LEN, 
        batch_size              = args.batch_size, 
        shuffle                 = True, 
        teacher_forcing         = False, 
        predict_eos_token       = False,
        truncate_method         = "head")

    validation_dataloader = PromptDataLoader(
        dataset                 = dataset["validation"], 
        template                = mytemplate, 
        tokenizer               = tokenizer,
        tokenizer_wrapper_class = WrapperClass, 
        max_seq_length          = MAX_LEN, 
        batch_size              = args.batch_size, 
        shuffle                 = True, 
        teacher_forcing         = False, 
        predict_eos_token       = False,
        truncate_method         = "head")

    test_dataloader = PromptDataLoader(
        dataset                 = dataset["test"], 
        template                = mytemplate, 
        tokenizer               = tokenizer,
        tokenizer_wrapper_class = WrapperClass, 
        max_seq_length          = MAX_LEN, 
        batch_size              = args.batch_size, 
        shuffle                 = True, 
        teacher_forcing         = False, 
        predict_eos_token       = False,
        truncate_method         = "head")

    return train_dataloader, validation_dataloader, test_dataloader



# ================ dataset selection ================
for data in datasets:
    args.data = data
    
    personalities = Personalities_dict[args.data]
    MAX_LEN       = MAX_LEN_dict[args.data]
    
    # ================ few shot number selection ================
    for shots in few_shot_list:
        args.shots = shots
        
        # ******** result name ********
        args.result_name  = './result/' + args.method + '_'+args.BASE + '_' + args.data + '_shots_' + str(args.shots) + \
                            '_Verbalizer_'+args.verbalizer+'_template_'+args.templates+'.txt'
        with open(args.result_name, 'w') as f:
            test_f1_total = []
            # ================ personality label selection ================
            for personality in personalities:
                args.personality = personality
                
                df_data = pd.read_csv( DATA_PATH_dict[args.data] + args.personality + '_whole.tsv', sep = '\t')
                df = df_data[['utterance', 'labels']]
                
                
                test_f1_all_seeds = []
                # ================ random seed selection ================
                for SEED in seeds:
                    args.SEED = SEED

                    np.random.seed(args.SEED)
                    torch.manual_seed(args.SEED)
                    torch.cuda.manual_seed_all(args.SEED)
                    torch.backends.cudnn.deterministic = True
                    os.environ["PYTHONHASHSEED"] = str(args.SEED)
                    torch.backends.cudnn.benchmark = False
                    torch.set_num_threads(1)

                    # ******** train test split ********
                    Uttr_train, Uttr_test, label_train, label_test = \
                        train_test_split(df['utterance'], df['labels'], test_size=0.1, random_state=SEED, stratify=df['labels'])

                    Uttr_train, Uttr_valid, label_train, label_valid = \
                        train_test_split(Uttr_train, label_train, test_size=0.1, random_state=SEED, stratify=label_train)

                    # ******** construct samples ********
                    dataset = {}
                    for split in ['train', 'validation', 'test']:
                        dataset[split] = []
                        cnt = 0
                        if split == 'train':
                            for u,l in zip(Uttr_train, label_train):
                                input_sample = InputExample(text_a=u, label=int(l),guid=cnt)
                                cnt += 1
                                dataset[split].append(input_sample)
                        elif split == 'validation':
                            for u,l in zip(Uttr_valid, label_valid):
                                input_sample = InputExample(text_a=u, label=int(l),guid=cnt)
                                cnt += 1
                                dataset[split].append(input_sample)
                        elif split == 'test':
                            for u,l in zip(Uttr_test, label_test):
                                input_sample = InputExample(text_a=u, label=int(l),guid=cnt)
                                cnt += 1
                                dataset[split].append(input_sample)
            

                    # ******** few shot setting ********
                    if args.shots > 0:
                        sampler = FewShotSampler(num_examples_per_label=args.shots, also_sample_dev=True, num_examples_per_label_dev=args.shots)
                        dataset['train'], dataset['validation'] = sampler(dataset['train'], seed=args.SEED)
                    elif args.shots == 0:
                        # no train
                        pass



                    # ***** construct verbalizer ***** 

                    with open('label_words/'+args.verbalizer+args.personality+'_words.txt', 'r') as f_verbalizer:
                            pos = [i.strip() for i in f_verbalizer.readline().split(',')][:100]
                            neg = [i.strip() for i in f_verbalizer.readline().split(',')][:80]

                    with open('label_words/'+args.verbalizer+args.personality+'_weights.txt', 'r') as f_verbalizer:
                            pos_weights = eval(f_verbalizer.readline())[:100]
                            neg_weights = eval(f_verbalizer.readline())[:80]


                    diff_len = len(neg_weights) - len(pos_weights)
                    
                    if diff_len >= 0:
                        label_words_weights = torch.Tensor([neg_weights, pos_weights + [0]*diff_len])
                    else:
                        label_words_weights = torch.Tensor([neg_weights + [0]*(-diff_len), pos_weights])
                    
                    print('label word weights: ', label_words_weights.shape)

                    # ***** template setting *****

                    candidate_templates = []
                    with open('templates/'+args.templates+args.personality+'_templates_top_10.txt', 'r') as f_template:
                        candidate_templates = [i.strip() for i in f_template.readlines()]
                    
                    logits_all_templates = []

                    plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-large")
                    wrapped_tokenizer = WrapperClass(max_seq_length=MAX_LEN, tokenizer=tokenizer,truncate_method="head")

                    

                    for template in candidate_templates:
                        
                        mytemplate = ManualTemplate(
                                text = template,
                                tokenizer = tokenizer,
                        )
                        
                        # ***** data loader ***** 
                    
                        train_dataloader, validation_dataloader, test_dataloader = data_loader(args, dataset, mytemplate, tokenizer, WrapperClass)
                        
                        class_labels = [0,1]
                        

                        myverbalizer = MyVerbalizer(
                            classes = class_labels,
                            label_words = {
                                0 : neg, 
                                1 : pos
                            },
                            tokenizer=tokenizer,
                            label_words_weights=label_words_weights.cuda())
                        
                        # ***** training setting *****
                        
                        prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
                        prompt_model = prompt_model.cuda()

                        loss_func = torch.nn.CrossEntropyLoss()
                        no_decay = ['bias', 'LayerNorm.weight']
                        
                        optimizer_grouped_parameters = [
                            {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                            {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                        ]

                        best_eval = 0
                        best_test = 0
                        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
                        
                        # ***** training process *****
                        logits_ = test(test_dataloader, prompt_model)
                        if args.shots != 0:
                            prompt_model.train()
                            for epoch in range(args.num_of_epoches):
                                tot_loss = 0
                                for step, inputs in enumerate(train_dataloader):

                                    inputs = inputs.cuda()
                                    
                                    logits = prompt_model(inputs)
                                    labels = inputs['label']
                                    loss   = loss_func(logits, labels)
                                    
                                    loss.backward()
                                    tot_loss += loss.item()
                                    optimizer.step()
                                    optimizer.zero_grad()
                                    if args.shots > 0:
                                        step_length = 1
                                    else:
                                        step_length = 10
                                    if step % step_length == 0: 
                                        eval_f1 = evaluation(validation_dataloader, prompt_model)
                                        if eval_f1 > best_eval:
                                            best_eval = eval_f1
                                            logits_ = test(test_dataloader, prompt_model)

                                        prompt_model.train()
                        else:
                            logits_ = test(test_dataloader, prompt_model)           
                        logits_all_templates.append(logits_.unsqueeze(dim=0))
                        
                    logits_all_templates = torch.cat(logits_all_templates)
                    print(logits_all_templates.shape)
                    logits_all_templates = torch.mean(logits_all_templates, 0).squeeze(0)
                    best_test = get_test_results(test_dataloader, logits_all_templates)
                    print('Current SEED:', args.SEED, 'TEST F1', best_test)
                    test_f1_all_seeds.append(best_test)

                test_f1_total.append(test_f1_all_seeds)
                print('\n========\n')
                print(test_f1_total)
            f.write(str(test_f1_total))






