"""
Produce a single, fully traceable "golden" output value from conv1 —
one MAC sequence, computed three different ways, that must all agree.
This is the reference value your first Verilog testbench checks against.

Why three ways?
  1. PyTorch float32 conv2d  -> sanity check against the real model
  2. Manual int8 MAC, done with plain Python ints/loops -> this is the
     exact sequence of multiply-accumulate operations your hardware
     will perform, written as readably as possible
  3. (printed) a step-by-step trace of every multiply, so you can
     follow along by hand if you want to double check method 2

If all three agree, you have a trustworthy golden value. Write it down
and use it as the first assertion in your conv1 MAC unit testbench.

Requires quantized_weights.json and model_fp32.pt from quantize.py/
train.py to already exist in this directory.

Usage:
    python verify_mac.py
"""

import json
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import TinyCNN


# Which output element to trace: output channel, row, col of conv1.
# Row 0 / col 0 (the image's top-left corner) is almost always pure
# black background in MNIST, since digits are centered with a border
# -- every input pixel there is 0, which makes every product 0 and
# "verifies" a MAC unit even if its multiplier is completely broken.
# Set these to None to auto-search for a patch that actually contains
# nonzero pixels (recommended), or set explicit ints to force a
# specific position.
OUT_CHANNEL = 0
OUT_ROW = None
OUT_COL = None


def find_nonzero_patch(image_q, out_h=26, out_w=26, min_nonzero=9):
    """
    Scan conv1's output positions and return a (row, col) whose 3x3
    input patch is well populated with nonzero pixels -- not just
    technically nonzero. A patch with only 1 of 9 pixels nonzero still
    leaves 8 of 9 multiplications as trivial 0*weight, which is a weak
    test for your MAC unit (it wouldn't catch a multiplier that's
    subtly wrong, only one that's completely broken).

    Tries to find a patch where all 9 pixels are nonzero first; if none
    exists, progressively relaxes the requirement and returns the
    patch with the most nonzero pixels found.
    """
    img = image_q[0]  # [28, 28]
    best_row, best_col, best_count = 0, 0, -1

    for target in range(min_nonzero, 0, -1):
        for row in range(out_h):
            for col in range(out_w):
                patch = img[row:row + 3, col:col + 3]
                count = int(torch.count_nonzero(patch))
                if count > best_count:
                    best_row, best_col, best_count = row, col, count
                if count >= target:
                    return row, col

    if best_count <= 0:
        print("WARNING: entire image is zero -- falling back to (0, 0).")
        return 0, 0
    return best_row, best_col


def load_quantized_weights(path="quantized_weights.json"):
    with open(path) as f:
        data = json.load(f)
    return data


def get_one_test_image():
    """Pull a single real MNIST test image, the same kind of input your
    hardware will eventually receive over UART."""
    transform = transforms.ToTensor()
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    image, label = test_set[0]  # first test image, shape [1, 28, 28]
    return image, label


def quantize_input_image(image, num_bits=8):
    """
    Quantize the input image the same way weights were quantized:
    symmetric per-tensor int8. Pixel values are already in [0, 1] from
    ToTensor(), so this maps [0, 1] -> [0, 127].
    """
    qmax = 2 ** (num_bits - 1) - 1
    max_val = image.abs().max().item()
    scale = max_val / qmax
    q = torch.round(image / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def method1_pytorch_float(image, out_channel, out_row, out_col):
    """Run the real trained model's conv1 in float32, return the one
    output value we're tracing."""
    model = TinyCNN()
    model.load_state_dict(torch.load("model_fp32.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        conv1_out = model.conv1(image.unsqueeze(0))  # [1, 4, 26, 26]
    value = conv1_out[0, out_channel, out_row, out_col].item()
    return value


def method2_manual_int8_mac(image_q, image_scale, weights_data, out_channel, out_row, out_col):
    """
    Compute the same conv1 output pixel using plain Python integer
    arithmetic on the quantized values — this is the literal sequence
    of operations your Verilog MAC unit will perform.

    conv1 has 1 input channel, 3x3 kernel, no padding, stride 1.
    Output[oc, oy, ox] = bias[oc] + sum over (ky, kx) of
                          input[oy+ky, ox+kx] * weight[oc, 0, ky, kx]

    All done in integer MAC operations, then rescaled back to a real
    number at the end using the scales (the hardware would instead
    pass the raw int32 accumulator forward, but we need a real number
    here to compare against PyTorch's float output).
    """
    w_info = weights_data["conv1.weight"]
    b_info = weights_data["conv1.bias"]

    w_shape = w_info["shape"]  # [4, 1, 3, 3]
    weights = w_info["values"]  # flat list, length 4*1*3*3 = 36
    weight_scale = w_info["scale"]

    bias_q = b_info["values"][out_channel]
    bias_scale = b_info["scale"]

    def weight_at(oc, ic, ky, kx):
        # flatten index into the [4, 1, 3, 3] tensor, row-major
        _, in_ch, kh, kw = w_shape
        idx = ((oc * in_ch + ic) * kh + ky) * kw + kx
        return weights[idx]

    image_list = image_q[0].tolist()  # [1, 28, 28] -> [28, 28] (single channel)

    acc_int = 0  # this is the integer accumulator, exactly as in hardware
    trace = []
    for ky in range(3):
        for kx in range(3):
            in_val = image_list[out_row + ky][out_col + kx]
            w_val = weight_at(out_channel, 0, ky, kx)
            product = in_val * w_val
            acc_int += product
            trace.append((ky, kx, in_val, w_val, product, acc_int))

    # The hardware adds the quantized bias directly to the int32
    # accumulator too, but the bias lives on a different scale than
    # the input*weight product (bias_scale vs image_scale*weight_scale).
    # In real hardware you'd requantize the bias to the accumulator's
    # scale before adding; here we do the equivalent by converting
    # everything to real numbers for the final comparison.
    acc_real = acc_int * image_scale * weight_scale
    bias_real = bias_q * bias_scale
    result = acc_real + bias_real

    return result, trace, acc_int, bias_q


def main():
    weights_data = load_quantized_weights()
    image, label = get_one_test_image()
    print(f"Using MNIST test image #0, true label = {label}")

    image_q, image_scale = quantize_input_image(image)

    out_row, out_col = OUT_ROW, OUT_COL
    if out_row is None or out_col is None:
        out_row, out_col = find_nonzero_patch(image_q)
        print(f"Auto-selected patch at row={out_row}, col={out_col} "
              f"(searched for the first 3x3 patch with nonzero pixels --")
        print(f"row 0/col 0 is almost always blank background in MNIST, "
              f"which would trivially zero out every product)")

    print(f"Tracing conv1 output[channel={OUT_CHANNEL}, row={out_row}, col={out_col}]\n")

    print(f"Input image quantization scale: {image_scale:.6f}")
    print(f"Quantized input range: [{image_q.min().item()}, {image_q.max().item()}]\n")

    fp32_value = method1_pytorch_float(image, OUT_CHANNEL, out_row, out_col)
    manual_value, trace, acc_int, bias_q = method2_manual_int8_mac(
        image_q, image_scale, weights_data, OUT_CHANNEL, out_row, out_col
    )

    print("Step-by-step MAC trace (method 2, integer domain):")
    print(f"{'ky':>3} {'kx':>3} {'input_q':>8} {'weight_q':>9} {'product':>8} {'running_acc':>12}")
    for ky, kx, in_val, w_val, product, running_acc in trace:
        print(f"{ky:>3} {kx:>3} {in_val:>8} {w_val:>9} {product:>8} {running_acc:>12}")
    print(f"\nFinal integer accumulator (sum of products): {acc_int}")
    print(f"Quantized bias value: {bias_q}")

    print("\n--- Comparison ---")
    print(f"Method 1 (PyTorch float32 conv1):      {fp32_value:.6f}")
    print(f"Method 2 (manual int8 MAC, rescaled):   {manual_value:.6f}")
    print(f"Absolute difference:                    {abs(fp32_value - manual_value):.6f}")

    print("\n--- Golden values for your Verilog testbench ---")
    print(f"Input pixels (row {out_row}-{out_row+2}, col {out_col}-{out_col+2}), int8:")
    for ky in range(3):
        row_vals = [image_q[0][out_row + ky][out_col + kx].item() for kx in range(3)]
        print(f"  {row_vals}")
    print(f"Weight kernel (channel {OUT_CHANNEL}), int8:")
    w_info = weights_data["conv1.weight"]
    for ky in range(3):
        row_vals = [
            w_info["values"][((OUT_CHANNEL * 1 + 0) * 3 + ky) * 3 + kx]
            for kx in range(3)
        ]
        print(f"  {row_vals}")
    print(f"Bias (channel {OUT_CHANNEL}), int8: {bias_q}")
    print(f"Expected raw accumulator (before bias, before ReLU): {acc_int}")
    print(f"Expected accumulator + bias (integer domain, bias requantized")
    print(f"  to match accumulator scale -- see note below): approx {acc_int} + rescaled bias")
    print("\nNote: in your Verilog testbench, feed exactly these 9 input")
    print("values and 9 weight values into your MAC unit and assert the")
    print(f"output equals {acc_int} before any bias or ReLU is applied.")
    print("That isolates the MAC array from the bias-add and activation")
    print("logic, which you should test as separate, simpler steps.")


if __name__ == "__main__":
    main()