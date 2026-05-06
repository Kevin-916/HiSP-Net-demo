import torch
from torch import nn

class ResMLP(nn.Module):
    def __init__(self, feature_num, cluster_num, nlayers, activation, dropout=0.0, use_ln=True):
        super().__init__()

        layers = []
        for _ in range(nlayers):
            layers.append(nn.Linear(feature_num, feature_num))
        self.layers = nn.ModuleList(layers)


        self.act = activation
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.ln = nn.LayerNorm(feature_num) if use_ln else nn.Identity()

        self.fc = nn.Linear(feature_num, cluster_num)

    def forward(self, x):
        res = x
        for i, layer in enumerate(self.layers):
            h = self.drop(self.act(layer(x)))
            res = res + h

        x = self.fc(x)
        return x


class ResMLP_2(nn.Module):
    def __init__(self, feature_num, cluster_num, nlayers,activation, dropout=0.0):
        super().__init__()
        self.act = activation
        layers = []
        for _ in range(nlayers):
            layers.append(nn.Linear(feature_num, feature_num))
        self.layers = nn.ModuleList(layers)

        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(feature_num)
        self.lns = nn.ModuleList([nn.LayerNorm(feature_num) for _ in range(nlayers)])
        self.fc = nn.Linear(feature_num, cluster_num)


    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.ln(h)
        for i, layer in enumerate(self.layers):
            residual = h
            h = self.lns[i](h)
            h = layer(h)
            h = self.drop(self.act(h))
            h = h + residual

        h = self.fc(h)
        return h