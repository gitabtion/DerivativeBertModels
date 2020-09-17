"""
@Time   :   2020-09-17 16:01:03
@File   :   constants.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PreTrainedModel:
    ELECTRA_SMALL_CHINESE = 'hfl/chinese-electra-small-discriminator'
