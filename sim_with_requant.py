"""
sim_with_requant.py

Simulates the full hardware integer pipeline WITH requantization,
to verify accuracy before implementing in SystemVerilog.

Run this on your machine:
    python sim_with_requant.py

It tests the first 20 MNIST test images and compares:
  1. Float32 Python model (baseline)
  2. Raw int8 pipeline WITHOUT requantization (what the hardware does now)
  3. Int8 pipeline WITH requantization (what we're about to implement)

The requantization operation is:
  output_q = clip(round(acc * MULT >> SHIFT), 0, 127)

where MULT and SHIFT come from requant_params.json.
"""

import json
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import TinyCNN
from quantize import QuantizedTinyCNN


def load_weights_int(path="quantized_weights.json"):
    with open(path) as f:
        data = json.load(f)
    w = {}
    for k in data:
        v = data[k]["values"]
        shape = data[k]["shape"]
        w[k] = torch.tensor(v, dtype=torch.float32).reshape(shape)
    return w


def requant(acc, mult, shift, relu=True):
    """Apply fixed-point requantization to a float tensor of accumulators."""
    q = torch.round(acc * mult / (2 ** shift))
    if relu:
        q = q.clamp(0, 127)
    else:
        q = q.clamp(-128, 127)
    return q


def run_raw_int8(image_q, weights):
    """Hardware pipeline WITHOUT requantization (current broken state)."""
    x = image_q.float().unsqueeze(0)

    w1 = weights["conv1.weight"]
    b1 = weights["conv1.bias"]
    x = F.relu(F.conv2d(x, w1, b1)).clamp(0, 127).round()

    x = F.max_pool2d(x, 2)

    w2 = weights["conv2.weight"]
    b2 = weights["conv2.bias"]
    x = F.relu(F.conv2d(x, w2, b2)).clamp(0, 127).round()

    x = F.max_pool2d(x, 2)

    x = x.flatten(1)
    wf = weights["fc.weight"]
    bf = weights["fc.bias"]
    x = F.relu(F.linear(x, wf, bf)).clamp(0, 127).round()

    return x.argmax().item()


def run_with_requant(image_q, weights, params):
    """Hardware pipeline WITH requantization (what we're implementing)."""
    x = image_q.float().unsqueeze(0)

    # Conv1
    w1 = weights["conv1.weight"]
    b1 = weights["conv1.bias"]
    acc = F.conv2d(x, w1, b1)
    m, s = params["conv1"]["multiplier"], params["conv1"]["shift"]
    x = requant(acc, m, s, relu=True)

    # Pool1 (scale-invariant, no requantization needed)
    x = F.max_pool2d(x, 2)

    # Conv2
    w2 = weights["conv2.weight"]
    b2 = weights["conv2.bias"]
    acc = F.conv2d(x, w2, b2)
    m, s = params["conv2"]["multiplier"], params["conv2"]["shift"]
    x = requant(acc, m, s, relu=True)

    # Pool2
    x = F.max_pool2d(x, 2)

    # FC
    x = x.flatten(1)
    wf = weights["fc.weight"]
    bf = weights["fc.bias"]
    acc = F.linear(x, wf, bf)
    m, s = params["fc"]["multiplier"], params["fc"]["shift"]
    x = requant(acc, m, s, relu=True)

    return x.argmax().item()


def run_float32(image, model):
    """Baseline float32 Python model."""
    with torch.no_grad():
        return model(image.unsqueeze(0)).argmax().item()


def main():
    # Load model
    model = TinyCNN()
    model.load_state_dict(torch.load("model_fp32.pt", map_location="cpu"))
    model.eval()

    # Load int8 weights
    weights = load_weights_int()

    # Load requant params
    with open("requant_params.json") as f:
        params = json.load(f)

    print(f"Requantization parameters:")
    for name, p in params.items():
        print(f"  {name}: mult={p['multiplier']}  shift={p['shift']}  "
              f"error={p['relative_error']*100:.4f}%")
    print()

    # Load MNIST test set
    transform = transforms.ToTensor()
    test_set = datasets.MNIST(root="./data", train=False,
                               download=True, transform=transform)

    # Test first 20 images
    N = 20
    results = {"float32": 0, "raw_int8": 0, "requant": 0}

    print(f"{'idx':>3}  {'label':>5}  {'float32':>7}  {'raw_int8':>8}  {'requant':>7}")
    print("-" * 40)

    for idx in range(N):
        image, label = test_set[idx]

        # Quantize input
        scale = image.abs().max().item() / 127
        image_q = torch.round(image / scale).clamp(-127, 127).to(torch.int8)

        fp32_pred   = run_float32(image, model)
        raw_pred    = run_raw_int8(image_q, weights)
        requant_pred = run_with_requant(image_q, weights, params)

        fp32_ok   = "✓" if fp32_pred   == label else "✗"
        raw_ok    = "✓" if raw_pred    == label else "✗"
        req_ok    = "✓" if requant_pred == label else "✗"

        if fp32_pred   == label: results["float32"]  += 1
        if raw_pred    == label: results["raw_int8"] += 1
        if requant_pred == label: results["requant"]  += 1

        print(f"{idx:>3}  {label:>5}  "
              f"{fp32_pred:>3}{fp32_ok:>4}  "
              f"{raw_pred:>4}{raw_ok:>4}  "
              f"{requant_pred:>3}{req_ok:>4}")

    print("-" * 40)
    print(f"{'Acc':>3}  {'':>5}  "
          f"{results['float32']:>3}/{N}{'':>4}  "
          f"{results['raw_int8']:>4}/{N}{'':>4}  "
          f"{results['requant']:>3}/{N}")
    print()
    print(f"Float32:          {results['float32']}/{N} = {results['float32']/N*100:.0f}%")
    print(f"Raw int8 (now):   {results['raw_int8']}/{N} = {results['raw_int8']/N*100:.0f}%")
    print(f"With requant:     {results['requant']}/{N} = {results['requant']/N*100:.0f}%")


if __name__ == "__main__":
    main()
