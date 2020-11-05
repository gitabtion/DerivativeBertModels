"""
@Time   :   2020-09-11 14:11:25
@File   :   dc_bert.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from derivative_bert_models.others.constants import PreTrainedModel, DEVICE
from .base_components import PositionalEncoding


class DcBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(PreTrainedModel.BERT_BASE_CHINESE)
        self.tokenizer = AutoTokenizer.from_pretrained(PreTrainedModel.BERT_BASE_CHINESE)
        trans_layer = nn.TransformerEncoderLayer(self.bert.config.hidden_size, 8)
        self.trans_encoder = nn.TransformerEncoder(trans_layer, 1)
        self.pe = PositionalEncoding(self.bert.config.hidden_size, 512)
        self.dense = nn.Linear(2 * self.bert.config.hidden_size, 2)

    def forward(self, x1, x2):
        # bert
        _tx1 = self.tokenizer(x1, padding=True, return_tensors='pt')
        _tx2 = self.tokenizer(x2, padding=True, return_tensors='pt')
        _tx1.to(DEVICE)
        _tx2.to(DEVICE)
        _x1 = self.bert(**_tx1)[0]
        _x2 = self.bert(**_tx2)[0]

        # get [cls] index in doc
        d_cls_id = _x1.shape[1]

        # concat to transformer
        x = torch.cat([_x1, _x2], dim=1)

        # positional encoding
        x = self.pe(x)

        # transformer
        x = self.trans_encoder(x)

        # get [cls] hidden states and concat
        x = torch.cat([x[:, 0, :], x[:, d_cls_id, :]], dim=-1)

        # using dense to compute score.
        out = self.dense(x)
        return out
