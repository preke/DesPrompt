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
import time




class ManualVerbalizer(Verbalizer):
    r"""
    The basic manually defined verbalizer class, this class is inherited from the :obj:`Verbalizer` class.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = True,
                 label_words_weights: Optional[List] = None):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.label_words = label_words
        self.label_words_weights = label_words_weights
        self.post_log_softmax = post_log_softmax

    def on_label_words_set(self):
        super().on_label_words_set()
        self.label_words = self.add_prefix(self.label_words, self.prefix)

         # TODO should Verbalizer base class has label_words property and setter?
         # it don't have label_words init argument or label words from_file option at all

        self.generate_parameters()

    @staticmethod
    def add_prefix(label_words, prefix):
        r"""Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.

        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]  #wrapped it to a list of list of label words.

        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        
        ## Tokenize 所有的 label words 得到 id
        
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)
        
        # [label_1_word_list, label_2_word_list, ...]
        # unsympathetic [8362, 5821, 8223, 9779, 9265] 一个词会被 tokenize 到多个 id (可能是wordpiece)

        
        max_len  = max([ max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])

        # print(max_len) # 最长的 label_words 的 wordpiece 长度
        
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        
        # print(max_num_label_words) # 225
        
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        
        
        
        
        # padding 成两个 mask 矩阵 for positive label words & negative label words
        words_ids_mask = [
            [[1]*len(ids) + [0]*(max_len-len(ids)) for ids in ids_per_label]
          + [[0]*max_len] * (max_num_label_words - len(ids_per_label))
                             for ids_per_label in all_ids
         ]
        
        # padding 成两个 word_id 矩阵 for positive label words & negative label words
        words_ids = [[ids + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label))
                             for ids_per_label in all_ids]

        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False) # A 3-d mask
        
        # words_ids_mask.shape: [batch_size * 2 (number_of_labels) * num_of_label_words * 7] 
        
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)
        
        # words_ids_mask.shape: [batch_size * 2 (number_of_labels) * num_of_label_words * 1] 
        
    def project(self,
                logits: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        r"""
        Project the labels, the return value is the normalized (sum to 1) probs of label words.

        Args:
            logits (:obj:`torch.Tensor`): The original logits of label words.

        Returns:
            :obj:`torch.Tensor`: The normalized logits of label words
        """

        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000*(1-self.label_words_mask)
        return label_words_logits

    def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        (4) Aggregate (for multiple label words)

        Args:
            logits (:obj:`torch.Tensor`): The original logits.

        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        # project
        label_words_logits = self.project(logits, **kwargs)  
        #Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)


        if self.post_log_softmax:
            # normalize
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            label_words_logits = torch.log(label_words_probs+1e-15)

        # aggregate
        label_logits = self.aggregate(label_words_logits)
        return label_logits

    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)


    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""Use weight to aggregate the logits of label words.

        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.

        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words.
        """
        
        '''
        label_words_logits.shape: [batch_size * num_of_labels* number_of_label_words]
        self.label_words_mask.shape [num_of_labels* number_of_label_words]
        
        (label_words_logits * self.label_words_mask).sum(-1)/self.label_words_mask.sum(-1)

        这一步 (label_words_logits * self.label_words_mask) 这里应该是对位相乘
        然后 .sum(-1) 是最后一维相加
        结合我们刚刚看到的shape 最后一维是 number_of_label_words， 所以这里应该是对所有的logits做了相加
        
        self.label_words_mask.sum(-1) 这一步的这个mask 我目前还没太搞懂是如何得到的，我输出我的样例来看，几乎是全1，所以这里看似是做了平均...
        
        下面是我输出的：
        self.label_words_mask
        
        tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')
        
        剩下的明天再探究了
        '''
        
        
        '''
        print('*'*100)
        print(label_words_logits)
        print(label_words_logits.shape) #[batch_size, 2, 225]
        print(self.label_words_mask)
        print(self.label_words_mask.shape) # [2, 225]
        print((label_words_logits * self.label_words_mask).sum(-1)) # tensor([[-1879.8622, -1884.2799]], device='cuda:0', grad_fn=<SumBackward1>)
        print(self.label_words_mask.sum(-1))        # tensor([204, 225], device='cuda:0')
        '''


#         label_words_logits = (label_words_logits * self.label_words_mask).sum(-1)/self.label_words_mask.sum(-1)
        # label_words_logits = (label_words_logits * self.label_words_weights).sum(-1)/self.label_words_weights.sum(-1)
        
        label_words_weights = F.softmax(self.label_words_weights-10000*(1-self.label_words_mask), dim=-1)
        label_words_logits = (label_words_logits * self.label_words_mask * label_words_weights).sum(-1)
        '''
        print(label_words_logits) # tensor([[-9.2150, -8.3746]],
        
        '''
        return label_words_logits

    def calibrate(self, label_words_probs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""

        Args:
            label_words_probs (:obj:`torch.Tensor`): The probability distribution of the label words with the shape of [``batch_size``, ``num_classes``, ``num_label_words_per_class``]

        Returns:
            :obj:`torch.Tensor`: The calibrated probability of label words.
        """
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() ==  1, "self._calibrate_logits are not 1-d tensor"
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] \
             and calibrate_label_words_probs.shape[0]==1, "shape not match"
        label_words_probs /= (calibrate_label_words_probs+1e-15)
        # normalize # TODO Test the performance
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1,keepdim=True) # TODO Test the performance of detaching()
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs



class KnowledgeableVerbalizer(ManualVerbalizer):
    r"""
    This is the implementation of knowledeagble verbalizer, which uses external knowledge to expand the set of label words.
    This class inherit the ``ManualVerbalizer`` class.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`classes`): The classes (or labels) of the current task.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer.
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        max_token_split (:obj:`int`, optional):
        verbalizer_lr (:obj:`float`, optional): The learning rate of the verbalizer optimization.
        candidate_frac (:obj:`float`, optional):
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer = None,
                 classes: Sequence[str] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 max_token_split: Optional[int] = -1,
                 verbalizer_lr: Optional[float]=5e-2,
                 candidate_frac: Optional[float]=0.5,
                 pred_temp: Optional[float]=1.0,
                 **kwargs):
        super().__init__(classes=classes, prefix=prefix, multi_token_handler=multi_token_handler, tokenizer=tokenizer, **kwargs)
        self.max_token_split = max_token_split
        self.verbalizer_lr = verbalizer_lr
        self.candidate_frac = candidate_frac
        self.pred_temp = pred_temp



    def on_label_words_set(self):
        self.label_words = self.delete_common_words(self.label_words)
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        self.generate_parameters()


    def delete_common_words(self, d):
        word_count = {}
        for d_perclass in d:
            for w in d_perclass:
                if w not in word_count:
                    word_count[w]=1
                else:
                    word_count[w]+=1
        for w in word_count:
            if word_count[w]>=2:
                for d_perclass in d:
                    if w in d_perclass[1:]:
                        findidx = d_perclass[1:].index(w)
                        d_perclass.pop(findidx+1)
        return d



    @staticmethod
    def add_prefix(label_words, prefix):
        r"""add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ' '.
        """
        new_label_words = []
        for words in label_words:
            new_label_words.append([prefix + word.lstrip(prefix) for word in words])
        return new_label_words




    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more one token.
        """
        all_ids = []
        label_words = []
        # print([len(x) for x in self.label_words], flush=True)
        for words_per_label in self.label_words:
            ids_per_label = []
            words_keep_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if self.max_token_split>0  and len(ids) > self.max_token_split:
                    # in knowledgebale verbalizer, the labelwords may be very rare, so we may
                    # want to remove the label words which are not recogonized by tokenizer.
                    logger.warning("Word {} is split into {} (>{}) tokens: {}. Ignored.".format(word, \
                                    len(ids), self.max_token_split,
                                    self.tokenizer.convert_ids_to_tokens(ids)))
                    continue
                else:
                    words_keep_per_label.append(word)
                    ids_per_label.append(ids)
            label_words.append(words_keep_per_label)
            all_ids.append(ids_per_label)
        self.label_words = label_words



        max_len  = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        words_ids_mask = [[[1]*len(ids) + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label))
                             for ids_per_label in all_ids]
        words_ids = [[ids + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label))
                             for ids_per_label in all_ids]

        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False) # A 3-d mask
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)
        self.label_words_weights = nn.Parameter(torch.zeros(self.num_classes, max_num_label_words), requires_grad=True)
        print("##Num of label words for each label: {}".format(self.label_words_mask.sum(-1).cpu().tolist()), flush=True)

        # print(self.label_words_ids.data.shape, flush=True)
        # print(self.words_ids_mask.data.shape, flush=True)
        # print(self.label_words_mask.data.shape, flush=True)
        # print(self.label_words_weights.data.shape, flush=True)
        # exit()
        # self.verbalizer_optimizer = torch.optim.AdamW(self.parameters(), lr=self.verbalizer_lr)




    def register_calibrate_logits(self, logits: torch.Tensor):
        r"""For Knowledgeable Verbalizer, it's nessessory to filter the words with has low prior probability.
        Therefore we re-compute the label words after register calibration logits.
        """
        if logits.requires_grad:
            logits = logits.detach()
        self._calibrate_logits = logits
        cur_label_words_ids = self.label_words_ids.data.cpu().tolist()
        print(self.candidate_frac, logits.shape[-1])
        rm_calibrate_ids = set(torch.argsort(self._calibrate_logits)[:int(self.candidate_frac*logits.shape[-1])].cpu().tolist())

        new_label_words = []
        for i_label, words_ids_per_label in enumerate(cur_label_words_ids):
            new_label_words.append([])
            print('For label ', i_label, ' ......\n')
            for j_word, word_ids in enumerate(words_ids_per_label):
                if j_word >= len(self.label_words[i_label]):
                    break
                if len((set(word_ids).difference(set([0]))).intersection(rm_calibrate_ids)) == 0:
                    new_label_words[-1].append(self.label_words[i_label][j_word])
                else:
                    print('Remove word ', self.label_words[i_label][j_word])
        self.label_words = new_label_words
        self.to(self._calibrate_logits.device)

    def project(self,
                 logits: torch.Tensor,
                 **kwargs,
                 ) -> torch.Tensor:
        r"""The return value if the normalized (sum to 1) probs of label words.
        """
        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000*(1-self.label_words_mask)
        return label_words_logits




    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""Use weight to aggregate the logots of label words.

        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.

        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words.
        """
        if not self.training:
            label_words_weights = F.softmax(self.pred_temp*self.label_words_weights-10000*(1-self.label_words_mask), dim=-1)
        else:
            label_words_weights = F.softmax(self.label_words_weights-10000*(1-self.label_words_mask), dim=-1)
        label_words_logits = (label_words_logits * self.label_words_mask * label_words_weights).sum(-1)
        return label_words_logits



    # def optimize(self,):
    #     self.verbalizer_optimizer.step()
    #     self.verbalizer_optimizer.zero_grad()


class MyVerbalizer(Verbalizer):
    r"""
    The basic manually defined verbalizer class, this class is inherited from the :obj:`Verbalizer` class.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = True,
                 label_words_weights: Optional[List] = None):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.label_words = label_words
        self.label_words_weights = label_words_weights
        self.post_log_softmax = post_log_softmax

    def on_label_words_set(self):
        super().on_label_words_set()
        self.label_words = self.add_prefix(self.label_words, self.prefix)

         # TODO should Verbalizer base class has label_words property and setter?
         # it don't have label_words init argument or label words from_file option at all

        self.generate_parameters()

    @staticmethod
    def add_prefix(label_words, prefix):
        r"""Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.

        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]  #wrapped it to a list of list of label words.

        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        
        ## Tokenize 所有的 label words 得到 id
        
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)[0] ## 只保留第一个
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)
        
        # [label_1_word_list, label_2_word_list, ...]
        # unsympathetic [8362, 5821, 8223, 9779, 9265] 一个词会被 tokenize 到多个 id (可能是wordpiece)

        
        max_len  = 1 ## 只保留第一个

        
        
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        
        # print(max_num_label_words) # 225
        
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        
        
        
        # padding 成两个 mask 矩阵 for positive label words & negative label words

        words_ids_mask = [
            [[1] for ids in ids_per_label]
          + [[0]] * (max_num_label_words - len(ids_per_label))
                             for ids_per_label in all_ids
         ]
        
        # padding 成两个 word_id 矩阵 for positive label words & negative label words
        words_ids = [[ids for ids in ids_per_label]
                             + [0]*(max_num_label_words-len(ids_per_label))
                             for ids_per_label in all_ids]

        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        



        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask  = nn.Parameter(words_ids_mask, requires_grad=False) # A 3-d mask
        
        # words_ids_mask.shape: [batch_size * 2 (number_of_labels) * num_of_label_words * 7] 
        
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)
        
        # words_ids_mask.shape: [batch_size * 2 (number_of_labels) * num_of_label_words * 1] 
        
    def project(self, logits: torch.Tensor, **kwargs,) -> torch.Tensor:
        r"""
        Project the labels, the return value is the normalized (sum to 1) probs of label words.

        Args:
            logits (:obj:`torch.Tensor`): The original logits of label words.

        Returns:
            :obj:`torch.Tensor`: The normalized logits of label words
        """


        
        n = 10
        top_n_values  = torch.topk(logits, n).values # [batch_size * n]
        top_n_indices = torch.topk(logits, n).indices # [batch_size * n]
        

        # 只适用于batch size == 1
        
        neg_match = list(set([int(i) for i in self.label_words_ids[0]]) & set([int(i) for i in top_n_indices[0]]))
        pos_match = list(set([int(i) for i in self.label_words_ids[1]]) & set([int(i) for i in top_n_indices[0]]))
        # print('Negative in top ', n ,' : ', neg_match)
        # print('Positive in top ', n ,' : ', pos_match)

        
        weights = [[],[]]
        for pos in pos_match:
            for i in range(len(self.label_words_ids[1])):
                if self.label_words_ids[1][i] == pos:
                    idx = i
                    break
            
            tmp = self.label_words_weights[1][idx]
            weights[1].append(tmp)
        for neg in neg_match:
            for i in range(len(self.label_words_ids[0])):
                if self.label_words_ids[0][i] == neg:
                    idx = i
                    break
            
            tmp = self.label_words_weights[0][idx]
            weights[0].append(tmp)

       

        max_num_label_words = 0
        if len(neg_match) > len(pos_match):
            max_num_label_words = len(neg_match)
        elif len(neg_match) < len(pos_match):
            max_num_label_words = len(pos_match)

        words_ids_mask = [
            [1 for ids in ids_per_label]
          + [0] * (max_num_label_words - len(ids_per_label))
                             for ids_per_label in [neg_match, pos_match]
        ]
        

        words_ids_mask = torch.tensor(words_ids_mask).cuda()
        self.label_words_mask = nn.Parameter(words_ids_mask, requires_grad=False)

        words_weights = [weights_per_label + [0] * (max_num_label_words - len(weights_per_label)) for weights_per_label in weights]
        words_weights = torch.tensor(words_weights).cuda()
        self.label_words_weights = words_weights
        

        if len(neg_match) > len(pos_match):
            pos_match = pos_match + [0] * (len(neg_match) - len(pos_match))
        elif len(neg_match) < len(pos_match):
            neg_match = neg_match + [0] * (len(pos_match) - len(neg_match))

        matched_label_words_ids = [neg_match, pos_match]
        self.label_words_ids = nn.Parameter(torch.tensor(matched_label_words_ids).cuda(), requires_grad=False)
        
        label_words_logits = logits[:, matched_label_words_ids]

        label_words_logits -= 10000*(1-self.label_words_mask)

        return label_words_logits

    def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        (4) Aggregate (for multiple label words)

        Args:
            logits (:obj:`torch.Tensor`): The original logits.

        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        # project
        label_words_logits = self.project(logits, **kwargs)  
        #Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)


        if self.post_log_softmax:
            # normalize
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            label_words_logits = torch.log(label_words_probs+1e-15)

        # aggregate


        
        
        label_logits = self.aggregate(label_words_logits)
        return label_logits

    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)


    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""Use weight to aggregate the logits of label words.

        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.

        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words.
        """
        
        
        # 在这里不做权重归一化，如果做了归一化体现不出top-n中recall个数的优越性
#         label_words_weights = F.softmax(self.label_words_weights-10000*(1-self.label_words_mask), dim=-1)
        label_words_weights = self.label_words_weights-10000*(1-self.label_words_mask)
        
        
        label_words_logits = (label_words_logits * self.label_words_mask * label_words_weights).sum(-1)
        
#         print(label_words_logits)
#         time.sleep(100)
        return label_words_logits

    def calibrate(self, label_words_probs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""

        Args:
            label_words_probs (:obj:`torch.Tensor`): The probability distribution of the label words with the shape of [``batch_size``, ``num_classes``, ``num_label_words_per_class``]

        Returns:
            :obj:`torch.Tensor`: The calibrated probability of label words.
        """
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() ==  1, "self._calibrate_logits are not 1-d tensor"
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] \
             and calibrate_label_words_probs.shape[0]==1, "shape not match"
        label_words_probs /= (calibrate_label_words_probs+1e-15)
        # normalize # TODO Test the performance
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1,keepdim=True) # TODO Test the performance of detaching()
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs







