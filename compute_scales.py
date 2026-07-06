"""
compute_scales.py

Computes fixed-point requantization parameters from your trained model's
quantization scales. Run this ONCE after quantize.py to generate
requant_params.json, which the hardware uses.

The math:
  Each layer computes acc = sum(input_q * weight_q) in raw integers.
  This represents a real value of acc * S_in * S_w.
  We want output_q in [0,127] with scale S_out, so:
    output_q = clip(round(acc * M), 0, 127)
  where M = S_in * S_w / S_out.

  We choose S_out so the theoretical maximum accumulator maps to 127:
    max_acc = num_terms * 127 * 127
    S_out = max_acc * S_in * S_w / 127

  This gives M = 1 / (num_terms * 127) for all layers.

  M is represented as (multiplier / 2^shift) for hardware.

Usage:
    python compute_scales.py
"""

import json
import math


def fixed_point_approx(M, max_shift=31):
    best = None
    best_err = float('inf')
    for shift in range(1, max_shift + 1):
        mult = round(M * (2 ** shift))
        if mult > 65535 or mult <= 0:
            continue
        approx = mult / (2 ** shift)
        err = abs(approx - M) / M
        if err < best_err:
            best_err = err
            best = (mult, shift, approx, err)
    return best


def main():
    with open("quantized_weights.json") as f:
        data = json.load(f)

    scales = {k: data[k]["scale"] for k in data}

    S_in_image = 0.007874   # canonical image quantization scale
                             # (from gen_input_image.py output)
    S_w1 = scales["conv1.weight"]
    S_w2 = scales["conv2.weight"]
    S_wf = scales["fc.weight"]

    # Compute output scales
    max_acc_conv1 = 9 * 127 * 127
    max_acc_conv2 = 36 * 127 * 127
    max_acc_fc    = 200 * 127 * 127

    S_out1 = (max_acc_conv1 * S_in_image * S_w1) / 127
    S_out2 = (max_acc_conv2 * S_out1 * S_w2) / 127
    S_outf = (max_acc_fc * S_out2 * S_wf) / 127

    M_conv1 = S_in_image * S_w1 / S_out1
    M_conv2 = S_out1 * S_w2 / S_out2
    M_fc    = S_out2 * S_wf / S_outf

    print("Requantization multipliers:")
    print(f"  M_conv1 = {M_conv1:.8f}  (= 1/{round(1/M_conv1):.0f})")
    print(f"  M_conv2 = {M_conv2:.8f}  (= 1/{round(1/M_conv2):.0f})")
    print(f"  M_fc    = {M_fc:.8f}  (= 1/{round(1/M_fc):.0f})")
    print()

    params = {}
    for name, M, nterms in [
        ("conv1", M_conv1, 9),
        ("conv2", M_conv2, 36),
        ("fc",    M_fc,    200),
    ]:
        mult, shift, approx, err = fixed_point_approx(M)
        params[name] = {
            "M": M,
            "multiplier": mult,
            "shift": shift,
            "approx": approx,
            "relative_error": err,
            "num_terms": nterms,
        }
        print(f"  {name}: M={M:.8f}  mult={mult}  shift={shift}  err={err*100:.4f}%")
        print(f"    Hardware: output_q = clip((acc * {mult}) >> {shift}, 0, 127)")

    with open("requant_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print("\nSaved: requant_params.json")
    print("Next step: run sim_with_requant.py to verify accuracy before hardware")


if __name__ == "__main__":
    main()
