"""
@Time   :   2020-11-04 12:05:48
@File   :   test_dc_bert.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""

import unittest
import torch
import torch.nn.functional as F
from derivative_bert_models.models import DcBert
from derivative_bert_models.others.constants import DEVICE


class DcBertTestCase(unittest.TestCase):
    def test_valid_model(self):
        text1 = ['测试', '测测']
        text2 = ['这是一个测试', '测试']
        model = DcBert()
        model.eval()
        print(model(text1, text2))

    def test_train_model(self):
        text1 = ['测试', '测测', '相关']
        text2 = ['这是一个测试', '测试', '不相干']
        labels = torch.tensor([1, 1, 0], dtype=torch.long)
        labels.to(DEVICE)
        model = DcBert()

        model.eval()
        print(model(text1, text2))

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # train step
        optimizer.zero_grad()
        out = model(text1, text2)
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()

        model.eval()
        print(model(text1, text2))
