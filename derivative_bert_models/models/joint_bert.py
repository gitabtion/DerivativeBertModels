"""
@Time   :   2020-11-04 15:17:54
@File   :   joint_bert.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from derivative_bert_models.others.constants import PreTrainedModel


class JointBert(nn.Module):
    """
    unofficial implementation of [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)
    """

    def __init__(self, num_classes, num_slots):
        super(JointBert, self).__init__()
        self.num_classes = num_classes
        self.bert = AutoModel.from_pretrained(PreTrainedModel.BERT_BASE_CHINESE)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, num_slots)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask, token_idx):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        intent_outputs = self.intent_classifier(self.dropout(bert_out[1]))
        hidden_states = torch.cat([])

        out = self.bert(**encoded_text).squeeze()
        hidden_state = self.attention(torch.softmax(hidden_state, dim=0)) + out[0][0]
        return self.intent_classifier(hidden_state), self.slot_classifier(out[0][1:-1]), hidden_state
