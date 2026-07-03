// weight_rom_tb.sv
//
// Verifies every address in the weight ROM and bias ROM returns the
// correct value. Checks all 36 weight entries and all 4 bias entries.
//
// The expected values here match the Python golden values from
// verify_mac.py -- channel 0 kernel must be [-16,54,101,59,85,126,11,39,5]
// and channel 0 bias must be -2.

`timescale 1ns / 1ps

module weight_rom_tb;

    logic [5:0]        weight_addr;
    logic signed [7:0] weight_out;
    logic [1:0]        bias_addr;
    logic signed [7:0] bias_out;

    weight_rom uut (
        .weight_addr(weight_addr),
        .weight_out (weight_out),
        .bias_addr  (bias_addr),
        .bias_out   (bias_out)
    );

    // Expected values: same flat ordering as quantized_weights.json
    // (oc*9 + ky*3 + kx), must match gen_weight_rom.py output exactly
    integer exp_weights [0:35];
    integer exp_biases  [0:3];
    integer errors;
    integer i;

    initial begin
        // oc=0
        exp_weights[ 0] = -16; exp_weights[ 1] =  54; exp_weights[ 2] = 101;
        exp_weights[ 3] =  59; exp_weights[ 4] =  85; exp_weights[ 5] = 126;
        exp_weights[ 6] =  11; exp_weights[ 7] =  39; exp_weights[ 8] =   5;
        // oc=1
        exp_weights[ 9] =   9; exp_weights[10] =  39; exp_weights[11] =  97;
        exp_weights[12] =  52; exp_weights[13] =  81; exp_weights[14] =  21;
        exp_weights[15] =  88; exp_weights[16] =  95; exp_weights[17] = -44;
        // oc=2
        exp_weights[18] = -70; exp_weights[19] = -41; exp_weights[20] =  -7;
        exp_weights[21] =  49; exp_weights[22] =  47; exp_weights[23] = 103;
        exp_weights[24] =  88; exp_weights[25] = 120; exp_weights[26] =  91;
        // oc=3
        exp_weights[27] =  63; exp_weights[28] =  67; exp_weights[29] = -56;
        exp_weights[30] =  66; exp_weights[31] = -45; exp_weights[32] = -88;
        exp_weights[33] = -62; exp_weights[34] =-127; exp_weights[35] =  16;

        exp_biases[0] =  -2;
        exp_biases[1] =  55;
        exp_biases[2] =  -3;
        exp_biases[3] = 127;

        errors = 0;

        // Check all 36 weight addresses
        $display("Checking all 36 weight ROM entries...");
        for (i = 0; i < 36; i = i + 1) begin
            weight_addr = i[5:0];
            #1; // combinational ROM -- just needs a tiny settling delay
            if (weight_out !== exp_weights[i]) begin
                $display("  FAIL addr=%0d: got %0d, expected %0d", i, weight_out, exp_weights[i]);
                errors = errors + 1;
            end
        end
        if (errors == 0)
            $display("  PASS: all 36 weights correct");

        // Extra spot-check: channel 0 kernel printed row by row
        $display("\nChannel 0 kernel spot-check (expect [-16,54,101 / 59,85,126 / 11,39,5]):");
        for (i = 0; i < 9; i = i + 1) begin
            weight_addr = i[5:0];
            #1;
            $write("  %4d", weight_out);
            if (i % 3 == 2) $display("");
        end

        // Check all 4 bias addresses
        $display("\nChecking all 4 bias ROM entries...");
        for (i = 0; i < 4; i = i + 1) begin
            bias_addr = i[1:0];
            #1;
            if (bias_out !== exp_biases[i]) begin
                $display("  FAIL bias oc=%0d: got %0d, expected %0d", i, bias_out, exp_biases[i]);
                errors = errors + 1;
            end
        end
        if (errors == 0)
            $display("  PASS: all 4 biases correct");

        $display("");
        if (errors == 0)
            $display("=== ALL TESTS PASSED ===");
        else
            $display("=== %0d TEST(S) FAILED ===", errors);

        $finish;
    end

endmodule