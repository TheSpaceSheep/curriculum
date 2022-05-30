# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any


class Receiver(nn.Module):
    def __init__(self, n_outputs, n_hidden):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(n_hidden, n_outputs)

    def forward(self, x, _input, _aux_input):
        return self.fc(x)

class Sender(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)

    def forward(self, x, _aux_input):
        return self.fc1(x)


class SenderWithEmbedding(nn.Module):
    def __init__(self, n_attributes, n_values, n_hidden, embed_dim=512):
        super(Sender, self).__init__()
        self.embedder = AttributeValueEmbedder(n_attributes, n_values, embed_dim)
        self.fc = nn.Linear(n_attributes*embed_dim, n_hidden)

    def forward(self, x, _aux_input):
        print(x.shape)
        out = self.embedder(x)
        print(out.shape)
        out = self.fc(out)
        print(out.shape)
        return out


class NonLinearReceiver(nn.Module):
    def __init__(self, n_outputs, vocab_size, n_hidden, max_length):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.fc_1 = nn.Linear(vocab_size * max_length, n_hidden)
        self.fc_2 = nn.Linear(n_hidden, n_outputs)

        self.diagonal_embedding = nn.Embedding(vocab_size, vocab_size)
        nn.init.eye_(self.diagonal_embedding.weight)

    def forward(self, x, _input, _aux_input):
        with torch.no_grad():
            x = self.diagonal_embedding(x).view(x.size(0), -1)

        x = self.fc_1(x)
        x = F.leaky_relu(x)
        x = self.fc_2(x)

        zeros = torch.zeros(x.size(0), device=x.device)
        return x, zeros, zeros


class Freezer(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.eval()

    def train(self, mode):
        pass

    def forward(self, *input):
        with torch.no_grad():
            r = self.wrapped(*input)
        return r


class PlusOneWrapper(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, *input):
        r1, r2, r3 = self.wrapped(*input)
        return r1 + 1, r2, r3


class AttributeValueEmbedder(nn.Module):
    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        embed_dim: int = 512,
        dropout: float = 0,
        freeze: bool = False,
        no_flatten: bool = False,
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.embed_dim = embed_dim
        self.freeze = freeze
        self.no_flatten = no_flatten
        self.embeddings = nn.Embedding(n_attributes * n_values, embed_dim)
        # self.embeddings = nn.Linear(n_attributes * n_values, embed_dim)
        # offsets to embed the various attributes
        offsets = torch.arange(n_attributes) * n_values
        self.register_buffer("attribute_offsets", offsets.unsqueeze(0))
        # Attribute dropout probability
        self.p_dropout = dropout
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.LongTensor, aux_input: Optional[Any] = None):
        bsz, *others = x.size()
        out = self.embeddings(x + self.attribute_offsets)
        
        if self.p_dropout > 0:
            out = self.dropout(out.unsqueeze(-1)).squeeze(-1)
        if self.freeze:
            out = out.detach()
        if not self.no_flatten:
            out = out.view(bsz, -1)
        return out
