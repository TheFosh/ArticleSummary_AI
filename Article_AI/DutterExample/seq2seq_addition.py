"""
CODE BY DR. DUTTER.
Should be used as a framework.
"""

import pickle
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from DutterExample import seq2seqAttn


class AdditionSet(Dataset):
    def __init__(self):
        with open("addition_dict.pkl", 'rb') as f:
            self.data_dict = pickle.load(f)
        self.keys_list = sorted(list(self.data_dict.keys()))
        self.num_keys = len(self.keys_list)
        self.current_key = 0

        self.current_inputs = None
        self.current_targets = None

        self.set_data()

    def set_data(self):
        ci, ct = zip(*self.data_dict[self.keys_list[self.current_key]])
        self.current_inputs = torch.tensor(ci, dtype=torch.long)
        self.current_targets = torch.tensor(ct, dtype=torch.long)

    def __len__(self):
        return len(self.current_inputs)

    def __getitem__(self, item):
        return self.current_inputs[item], self.current_targets[item]

def train_nn(epochs=5, batch_size=32, lr=0.001):
    addition_data = AdditionSet()

    model = seq2seqAttn.Seq2Seq(13, 13, 32, 32, 128, 2, 11, 12)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        indices = list(range(addition_data.num_keys))
        if epoch > 0:
            random.shuffle(indices)
        progress_bar = tqdm(range(addition_data.num_keys), desc=f"Epoch {epoch + 1}/{epochs}")
        for i, _ in enumerate(progress_bar):
            addition_data.current_key = indices[i]
            addition_data.set_data()
            addition_loader = DataLoader(addition_data, batch_size=batch_size, shuffle=True)
            for data in addition_loader:
                inputs, targets = data

                optimizer.zero_grad()
                output = model(inputs, targets, 0.0)
                output = output.view(-1, output.size(-1))
                target = targets.view(-1)

                loss = loss_fn(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                progress_bar.set_postfix({'loss': loss.item(), 'size': addition_data.keys_list[indices[i]]})
        torch.save(model.state_dict(), "addition.pt")


def add_numbers():
    model = seq2seqAttn.Seq2Seq(13, 13, 32, 32, 128, 2, 11, 12)
    model.load_state_dict(torch.load("addition.pt", map_location="cpu"))
    model.eval()
    input_str = input("Enter the first number: ")
    input_str_2 = input("Enter the second number: ")
    input_tensor = torch.tensor(
        [11] + [int(char) for char in input_str] + [10] + [int(char) for char in input_str_2] + [12],
        dtype=torch.long).view(1, -1)
    print(model.predict(input_tensor)[0])


add_numbers()
