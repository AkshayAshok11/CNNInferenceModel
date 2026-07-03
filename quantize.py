"""
Quantize TinyCNN's float32 weights to 8-bit fixed-point integers, and
verify accuracy doesn't collapse when running inference with the
quantized values. This is the bridge between Python and Verilog: the
numbers this script produces are exactly what your MAC unit testbenches
will use as inputs.

Quantization scheme: per-tensor symmetric int8 quantization.
    scale = max(abs(tensor)) / 127
    q = round(tensor / scale), clipped to [-127, 127]
    dequantized = q * scale  (this is what the simulated quantized
                               forward pass uses, to predict the
                               accuracy hit before you ever touch RTL)

This is the simplest quantization scheme that exists, deliberately.
Per-channel quantization or learned scales give better accuracy, but
add complexity to the hardware (separate scale per output channel
means separate scaling logic per MAC lane). Start simple, get the full
hardware pipeline working, then revisit if accuracy is too low.

Usage:
    python quantize.py
"""

import json
import torch
import torch.nn.functional as F

from model import TinyCNN
from train import get_dataloaders, evaluate


def quantize_tensor(tensor, num_bits=8):
    """Symmetric per-tensor quantization. Returns (int8 tensor, scale)."""
    qmax = 2 ** (num_bits - 1) - 1  # 127 for int8
    max_val = tensor.abs().max().item()
    if max_val == 0:
        max_val = 1e-8  # avoid div by zero on an all-zero tensor (e.g. some biases)
    scale = max_val / qmax
    q = torch.round(tensor / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_tensor(q_tensor, scale):
    return q_tensor.to(torch.float32) * scale


class QuantizedTinyCNN:
    """
    Runs inference using dequantized (but originally int8) weights, to
    measure the accuracy impact of quantization before any hardware
    exists. This mimics what your fixed-point MAC pipeline will compute,
    just executed in float32 on the CPU for fast iteration.
    """

    def __init__(self, state_dict):
        self.q_weights = {}
        self.scales = {}
        for name, tensor in state_dict.items():
            q, scale = quantize_tensor(tensor)
            self.q_weights[name] = q
            self.scales[name] = scale

    def get_dequantized(self, name):
        return dequantize_tensor(self.q_weights[name], self.scales[name])

    def forward(self, x):
        w1 = self.get_dequantized("conv1.weight")
        b1 = self.get_dequantized("conv1.bias")
        w2 = self.get_dequantized("conv2.weight")
        b2 = self.get_dequantized("conv2.bias")
        wf = self.get_dequantized("fc.weight")
        bf = self.get_dequantized("fc.bias")

        x = F.relu(F.conv2d(x, w1, b1))
        x = F.max_pool2d(x, 2)
        x = F.relu(F.conv2d(x, w2, b2))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.linear(x, wf, bf)
        return x

    def export_for_verilog(self, path="quantized_weights.json"):
        """
        Dump every quantized tensor and its scale to JSON, in a format
        your Verilog testbenches can read directly (as $readmemh-style
        hex, or parsed by a small script into .mem files).
        """
        export = {}
        for name, q in self.q_weights.items():
            export[name] = {
                "shape": list(q.shape),
                "scale": self.scales[name],
                # flatten to a plain list of ints for easy JSON/hex export
                "values": q.flatten().tolist(),
            }
        with open(path, "w") as f:
            json.dump(export, f, indent=2)
        print(f"Exported quantized weights to {path}")


def evaluate_quantized(qmodel, loader, device):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = qmodel.forward(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


if __name__ == "__main__":
    device = torch.device("cpu")  # quantized inference is fast, CPU is fine

    model = TinyCNN()
    model.load_state_dict(torch.load("model_fp32.pt", map_location=device))
    model.eval()

    _, test_loader = get_dataloaders()

    fp32_acc = evaluate(model, test_loader, device)
    print(f"Float32 baseline accuracy: {fp32_acc*100:.2f}%")

    qmodel = QuantizedTinyCNN(model.state_dict())
    quant_acc = evaluate_quantized(qmodel, test_loader, device)
    print(f"Int8 quantized accuracy:   {quant_acc*100:.2f}%")
    print(f"Accuracy drop:             {(fp32_acc - quant_acc)*100:.2f} points")

    print("\nPer-tensor scales (you'll need these for the hardware's")
    print("output rescaling logic between layers):")
    for name, scale in qmodel.scales.items():
        print(f"  {name:20s} scale={scale:.6f}")

    qmodel.export_for_verilog()
