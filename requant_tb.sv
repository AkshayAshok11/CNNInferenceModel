// requant_tb.sv
//
// Testbench for the requant module, testing all three layer
// parameterizations (conv1, conv2, fc) against golden values
// computed by sim_with_requant.py.

`timescale 1ns / 1ps

module requant_tb;

    // We test three instances with different parameters
    logic signed [19:0] acc_conv1, acc_conv2, acc_fc;
    logic signed [7:0]  out_conv1, out_conv2, out_fc;

    requant #(.MULT(33691), .SHIFT(24)) rq_conv1 (
        .acc(acc_conv1), .out_q(out_conv1));

    requant #(.MULT(38683), .SHIFT(25)) rq_conv2 (
        .acc(acc_conv2), .out_q(out_conv2));

    requant #(.MULT(30751), .SHIFT(25)) rq_fc (
        .acc(acc_fc), .out_q(out_fc));

    integer errors;

    task check(
        input logic signed [7:0] got,
        input integer             expected,
        input string              label
    );
        if (got !== expected[7:0]) begin
            $display("  FAIL %s: got %0d, expected %0d", label, got, expected);
            errors = errors + 1;
        end else begin
            $display("  PASS %s: %0d", label, got);
        end
    endtask

    initial begin
        errors = 0;

        // --- Conv1 (MULT=33691, SHIFT=24) ---
        $display("--- conv1 requant (mult=33691 shift=24) ---");

        acc_conv1 = 20'sd48387;   #1;
        check(out_conv1, 97, "acc=48387 -> 97 (golden conv1 pixel)");

        acc_conv1 = 20'sd0;       #1;
        check(out_conv1, 0, "acc=0 -> 0");

        acc_conv1 = -20'sd10000;  #1;
        check(out_conv1, 0, "acc=-10000 -> 0 (ReLU)");

        // Near theoretical max -- use max signed 20-bit value
        acc_conv1 = 20'sd524287;  #1; // 2^19 - 1
        check(out_conv1, 127, "acc=max -> 127 (saturation)");

        // --- Conv2 (MULT=38683, SHIFT=25) ---
        $display("--- conv2 requant (mult=38683 shift=25) ---");

        acc_conv2 = 20'sd3613;    #1;
        check(out_conv2, 4, "acc=3613 -> 4 (golden conv2 pixel)");

        acc_conv2 = 20'sd0;       #1;
        check(out_conv2, 0, "acc=0 -> 0");

        acc_conv2 = -20'sd50000;  #1;
        // -50000 doesn't fit in 20-bit signed (range -524288..524287)
        // use a value that does: -20'sd50000 wraps, use -20'sd32768
        acc_conv2 = -20'sd32768;  #1;
        check(out_conv2, 0, "acc negative -> 0 (ReLU)");

        acc_conv2 = 20'sd110163;  #1;
        check(out_conv2, 127, "acc=110163 -> 127 (calibrated max)");

        // --- FC (MULT=30751, SHIFT=25) ---
        $display("--- fc requant (mult=30751 shift=25) ---");

        acc_fc = 20'sd0;          #1;
        check(out_fc, 0, "acc=0 -> 0");

        acc_fc = -20'sd5000;      #1;
        check(out_fc, 0, "acc=-5000 -> 0 (ReLU)");

        acc_fc = 20'sd138577;     #1;
        check(out_fc, 126, "acc=138577 -> 126 (calibrated max)");

        acc_fc = 20'sd69000;      #1;
        check(out_fc, 63, "acc=69000 -> 63 (half max)");

        $display("");
        if (errors == 0)
            $display("=== ALL TESTS PASSED ===");
        else
            $display("=== %0d TEST(S) FAILED ===", errors);

        $finish;
    end

endmodule
