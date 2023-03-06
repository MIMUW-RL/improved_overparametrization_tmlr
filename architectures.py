import torch
from torch import nn
import numpy as np
from numpy import random


class BasicNet1L(nn.Module):
    def __init__(self, d0: int, d1: int, d2: int, device="cuda"):
        super().__init__()
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.W = nn.Linear(d0, d1, bias=False)

        with torch.no_grad():
            # W iid N(0,1)
            self.W.weight = torch.nn.Parameter(torch.normal(mean=torch.zeros(d1, d0), std=1.0))
            signs = torch.ones(self.d1 * self.d2)
            idx = list(range(self.d1 * self.d2))
            random.shuffle(idx)
            signs[idx[: int((self.d1 * self.d2) / 2)]] = -1
            # V = +/- sqrt(|y|/Nd1) (połowa + połowa-)
            self.V = (torch.reshape(signs, (self.d2, self.d1)) * 1.0 / np.sqrt(self.d1)).to(device=device)

    def forward(self, x):
        self.preH = self.W(x)
        self.H = torch.relu(self.W(x))
        out = self.H @ self.V.T
        return out


class BasicNet2L(nn.Module):
    def __init__(self, d0: int, d1: int, d2: int, device="cuda"):
        super().__init__()
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.W = nn.Linear(d0, d1, bias=False)
        self.V = nn.Linear(d1, d2, bias=False)

        with torch.no_grad():
            self.W.weight = torch.nn.Parameter(torch.normal(mean=torch.zeros(d1, d0), std=1.0))
            signs = torch.ones(self.d1 * self.d2)
            idx = list(range(self.d1 * self.d2))
            random.shuffle(idx)
            signs[idx[: int((self.d1 * self.d2) / 2)]] = -1
            # V = +/- sqrt(|y|/Nd1) (połowa + połowa-)
            # self.V.weight = torch.nn.Parameter((torch.reshape(signs, (self.d2, self.d1)) * 1.0 / np.sqrt(self.d1)).to(device=device))

    def forward(self, x):
        self.preH = self.W(x)
        self.H = torch.relu(self.W(x))
        out = self.V(self.H)
        return out
