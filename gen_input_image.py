"""
gen_input_image.py

Loads MNIST test image #0 (the "7" we've been using throughout),
quantizes it to int8 the same way quantize.py does, and exports it
in two formats:

  input_image.hex  -- 784 lines, one byte per line in hex, for use
                      with Verilog's $readmemh system task. This is
                      how the input buffer loads the image at the
                      start of simulation.

  input_image.txt  -- human-readable grid (28x28) so you can see
                      which pixels are nonzero and visually confirm
                      the digit shape looks like a 7.

The quantization here must exactly match quantize.py's scheme:
  scale = max(abs(image)) / 127
  q = round(image / scale), clipped to [-127, 127]

Since MNIST pixels are in [0,1] (all non-negative), every quantized
value will be in [0, 127] -- no negative pixel values. This means
the input buffer only needs to handle unsigned values in practice,
but we store them as signed int8 to keep the MAC unit's arithmetic
consistent (int8 * int8 signed multiply).

Usage:
    python gen_input_image.py
    (run from the same folder as your other project files)
"""

import torch
from torchvision import datasets, transforms


def quantize_tensor(tensor, num_bits=8):
    """Identical to the function in quantize.py -- must stay in sync."""
    qmax = 2 ** (num_bits - 1) - 1  # 127
    max_val = tensor.abs().max().item()
    if max_val == 0:
        max_val = 1e-8
    scale = max_val / qmax
    q = torch.round(tensor / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def export_hex(q_image, path="input_image.hex"):
    """
    Write 784 lines of hex, one byte per line, row-major.
    Verilog's $readmemh loads these directly into a memory array.

    Signed int8 values are written as their two's-complement 8-bit
    representation so $readmemh interprets them correctly:
      positive values: just the hex (e.g. 7f for 127)
      negative values: two's complement (e.g. ff for -1, fe for -2)
    """
    pixels = q_image[0].flatten().tolist()  # [784] flat list
    with open(path, "w") as f:
        for val in pixels:
            # Convert to unsigned 8-bit two's complement
            byte = val & 0xFF
            f.write(f"{byte:02x}\n")
    print(f"Written: {path}  ({len(pixels)} pixels)")


def export_txt(q_image, scale, label, path="input_image.txt"):
    """Human-readable 28x28 grid for visual sanity check."""
    img = q_image[0]  # [28, 28]
    lines = [
        f"MNIST test image #0, true label = {label}",
        f"Quantization scale = {scale:.6f}",
        f"Pixel values are int8 (0..127 for this image, since all pixels >= 0)",
        f"Nonzero pixels show the digit shape; zeros are background.",
        "",
        "Quantized pixel grid (28x28, row-major):",
        "  (each cell is the int8 value; '  .' means zero/background)",
        "",
    ]
    for row in range(28):
        row_str = ""
        for col in range(28):
            val = img[row][col].item()
            if val == 0:
                row_str += "  ."
            else:
                row_str += f"{val:3d}"
        lines.append(row_str)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {path}")


def main():
    transform = transforms.ToTensor()
    test_set = datasets.MNIST(root="./data", train=False,
                               download=True, transform=transform)
    image, label = test_set[0]
    print(f"Loaded MNIST test image #0, true label = {label}")

    q_image, scale = quantize_tensor(image)
    print(f"Quantization scale: {scale:.6f}")
    print(f"Pixel range after quantization: [{q_image.min().item()}, {q_image.max().item()}]")

    # Sanity check: the top-left patch should still be all zeros
    # (same observation as verify_mac.py -- background border)
    patch = q_image[0][0:3, 0:3]
    print(f"Top-left 3x3 patch (should be all 0s -- background): {patch.tolist()}")

    # And the patch we verified earlier (row=7, col=6) should be nonzero
    patch2 = q_image[0][7:10, 6:9]
    print(f"Verified patch (row=7,col=6, should match verify_mac golden inputs):")
    print(f"  {patch2.tolist()}")
    golden = [[42, 92, 79], [111, 127, 127], [33, 57, 36]]
    if patch2.tolist() == golden:
        print("  PASS: matches verify_mac.py golden input pixels exactly")
    else:
        print("  NOTE: values differ from golden -- your model may have been retrained")

    export_hex(q_image)
    export_txt(q_image, scale, label)


if __name__ == "__main__":
    main()
