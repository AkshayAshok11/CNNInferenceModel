"""
Tiny CNN for MNIST, sized deliberately small so it maps cleanly to an FPGA.

Architecture (kept intentionally simple — every layer here becomes a
hardware module later):
    Input:    1x28x28 grayscale image
    Conv1:    4 filters, 3x3, stride 1, no padding  -> 4x26x26
    ReLU
    MaxPool:  2x2                                    -> 4x13x13
    Conv2:    8 filters, 3x3, stride 1, no padding   -> 8x11x11
    ReLU
    MaxPool:  2x2                                    -> 8x5x5  (floor)
    Flatten:  8*5*5 = 200
    FC:       200 -> 10

Why so small? An 8-bit MAC array on a Basys 3 (Artix-7, ~33k logic cells)
cannot fit a deep network's worth of parallel multipliers and on-chip
weight storage. Every parameter here lives in block RAM, and the layer
count is small enough that you can implement and debug each one as a
separate, fully-verified Verilog module before composing them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, bias=True)
        self.fc = nn.Linear(8 * 5 * 5, 10, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = TinyCNN()
    print(f"Total parameters: {model.param_count()}")
    dummy = torch.randn(1, 1, 28, 28)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
