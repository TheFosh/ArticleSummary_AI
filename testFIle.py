import torch
from torch import nn
from torch.utils.data import DataLoader

from cleanarticle import TestArticleData

dataset = TestArticleData()
article_loader = DataLoader(dataset, batch_size=1, shuffle= False)

for data in article_loader:
    inputs, targets = data
    print(inputs)
    print(targets)