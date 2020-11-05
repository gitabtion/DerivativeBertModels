"""
@Time   :   2020-11-04 14:52:52
@File   :   sentence_bert.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from derivative_bert_models.others.constants import PreTrainedModel, DEVICE


class SentenceBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(PreTrainedModel.BERT_BASE_CHINESE)
        self.tokenizer = AutoTokenizer.from_pretrained(PreTrainedModel.BERT_BASE_CHINESE)
        self.dense = nn.Linear(3 * self.bert.config.hidden_size, 2)

    def forward(self, x1, x2):
        # bert
        _tx1 = self.tokenizer(x1, padding=True, return_tensors='pt')
        _tx2 = self.tokenizer(x2, padding=True, return_tensors='pt')
        _tx1.to(DEVICE)
        _tx2.to(DEVICE)
        u = self.bert(**_tx1)[1]
        v = self.bert(**_tx2)[1]

        # concat the hidden states
        x = torch.cat([u, v, torch.abs(u - v)], dim=-1)

        # using dense to compute score.
        out = self.dense(x)
        return out
