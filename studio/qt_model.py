"""Train a classical LSTM through a differentiable quantum state-vector generator."""
import math

import torch
from torch import nn


class Forecast(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, inputs):
        sequence, _ = self.lstm(inputs)
        return self.head(sequence[:, -1]).squeeze(-1)


class QuantumWeights(nn.Module):
    """Two layers of RY rotations and a nearest-neighbor CNOT ring."""
    def __init__(self, size, depth=2, width=24):
        super().__init__()
        self.size = size
        self.qubits = (size - 1).bit_length()
        dim = 1 << self.qubits
        index = torch.arange(dim)
        bits = (index[:, None] >> torch.arange(self.qubits)) & 1
        self.register_buffer("bits", bits[:size].float())
        destination = index.clone()
        for control in range(self.qubits):
            target = (control + 1) % self.qubits
            destination ^= ((destination >> control) & 1) << target
        self.register_buffer("permutation", destination.argsort())
        initial = torch.zeros(dim)
        initial[0] = 1
        self.register_buffer("initial", initial)
        self.angles = nn.Parameter(torch.randn(depth, self.qubits) * 0.15 + math.pi / 4)
        self.mapper = nn.Sequential(nn.Linear(self.qubits + 1, width), nn.Tanh(), nn.Linear(width, 1))

    def probabilities(self):
        state = self.initial
        for layer in self.angles:
            for qubit, angle in enumerate(layer):
                zero, one = state.reshape(-1, 2, 1 << qubit).unbind(1)
                c, s = (angle / 2).cos(), (angle / 2).sin()
                state = torch.stack((c * zero - s * one, s * zero + c * one), 1).flatten()
            state = state[self.permutation]
        return state.square()

    def forward(self):
        probs = self.probabilities()
        # Scale by the state count so a uniform-probability input has magnitude 1.
        coordinates = torch.cat((self.bits, probs[:self.size, None] * probs.numel()), 1)
        return self.mapper(coordinates).flatten()


def generated_parameters(target, generator):
    flat = generator()
    sizes = [p.numel() for p in target.parameters()]
    return {name: values.reshape_as(param) for (name, param), values in
            zip(target.named_parameters(), flat.split(sizes), strict=True)}


class SoftmaxWeights(QuantumWeights):
    def __init__(self, size, depth, width):
        super().__init__(size, depth, width)
        del self.angles
        self.logits = nn.Parameter(torch.zeros(1 << self.qubits))

    def probabilities(self):
        return self.logits.softmax(dim=0)
