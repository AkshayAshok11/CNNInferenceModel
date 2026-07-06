// conv2_tb.sv
//
// Testbench for conv2 (8 output channels, 4 input channels, 3x3 kernel).
// Uses a synthetic pool1 buffer: value = flat_index % 127
//
// Tests:
//   1. oc=0, oy=0, ox=0: acc=-55445, bias=14 -> ReLU -> 0   (addr=0)
//      Verifies ReLU clips negative values to zero correctly.
//   2. oc=1, oy=7, ox=9: acc=3613, bias=-8 -> 3605 -> clip -> 127 (addr=207)
//      Verifies positive saturation and correct 36-step accumulation.
//   3. All 968 output pixels written (coverage).
//   4. All values in [0,127] (ReLU working).

`timescale 1ns / 1ps

module conv2_tb;

    logic        clk, rst, start, done;
    logic [9:0]  in_addr;
    logic signed [7:0] in_data;
    logic        out_wen;
    logic [9:0]  out_addr;
    logic signed [7:0] out_data;

    // Synthetic pool1 buffer: value = flat_index % 127
    logic signed [7:0] pool_buf [0:675];
    always_ff @(posedge clk)
        in_data <= pool_buf[in_addr];

    conv2 dut (
        .clk    (clk),  .rst    (rst),
        .start  (start),.done   (done),
        .in_addr(in_addr), .in_data(in_data),
        .out_wen(out_wen), .out_addr(out_addr), .out_data(out_data)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // Capture output writes
    logic signed [7:0] output_buf [0:967];
    always_ff @(posedge clk)
        if (out_wen) output_buf[out_addr] <= out_data;

    integer i, errors, cycle_count;

    initial begin
        errors = 0; cycle_count = 0;

        // Fill pool1 buffer with synthetic pattern
        for (i = 0; i < 676; i = i + 1)
            pool_buf[i] = i % 127;

        // Initialize output sentinel
        for (i = 0; i < 968; i = i + 1)
            output_buf[i] = -1;  // sentinel

        rst <= 1; start <= 0;
        repeat(3) @(posedge clk); #1;
        rst <= 0; @(posedge clk); #1;

        $display("Starting conv2 inference (8x11x11 output = 968 pixels)...");
        start <= 1; @(posedge clk); #1; start <= 0;

        while (!done) begin
            @(posedge clk); #1;
            cycle_count = cycle_count + 1;
            if (cycle_count > 200000) begin
                $display("FAIL: timeout"); errors = errors + 1; $finish;
            end
        end
        // Wait one extra cycle for final write to settle into output_buf
        @(posedge clk); #1;
        $display("done after %0d cycles", cycle_count);

        // Test 1: ReLU clipping (oc=0, oy=0, ox=0 -> expect 0)
        $display("");
        $display("--- Test 1: ReLU clips negative (addr=0, expect 0) ---");
        $display("    acc=-55445 + bias=14 = -55431 -> ReLU -> 0");
        if (output_buf[0] === 8'sd0)
            $display("PASS: output_buf[0] = 0");
        else begin
            $display("FAIL: output_buf[0] = %0d (expected 0)", output_buf[0]);
            errors = errors + 1;
        end

        // Test 2: positive saturation (oc=1, oy=7, ox=9 -> expect 127)
        $display("");
        $display("--- Test 2: 36-step accumulation + clip (addr=207, expect 127) ---");
        $display("    acc=3613 + bias=-8 = 3605 -> clip -> 127");
        if (output_buf[207] === 8'sd127)
            $display("PASS: output_buf[207] = 127");
        else begin
            $display("FAIL: output_buf[207] = %0d (expected 127)", output_buf[207]);
            errors = errors + 1;
        end

        // Test 3: all 968 pixels written
        $display("");
        $display("--- Test 3: all 968 outputs written ---");
        begin
            integer unwritten;
            unwritten = 0;
            for (i = 0; i < 968; i = i + 1)
                if (output_buf[i] === -1) unwritten = unwritten + 1;
            if (unwritten == 0)
                $display("PASS: all 968 pixels written");
            else begin
                $display("FAIL: %0d unwritten", unwritten);
                errors = errors + 1;
            end
        end

        // Test 4: no negative values
        $display("");
        $display("--- Test 4: all values in [0,127] ---");
        begin
            integer neg_count;
            neg_count = 0;
            for (i = 0; i < 968; i = i + 1)
                if (output_buf[i] < 0) neg_count = neg_count + 1;
            if (neg_count == 0)
                $display("PASS: all values >= 0");
            else begin
                $display("FAIL: %0d negative values", neg_count);
                errors = errors + 1;
            end
        end

        // Summary: nonzero pixels per channel
        $display("");
        $display("--- Nonzero pixel count per output channel ---");
        begin
            integer ch, nz;
            for (ch = 0; ch < 8; ch = ch + 1) begin
                nz = 0;
                for (i = 0; i < 121; i = i + 1)
                    if (output_buf[ch*121 + i] !== 8'sd0) nz = nz + 1;
                $display("  Channel %0d: %0d / 121 nonzero (%.1f%%)",
                         ch, nz, nz * 100.0 / 121.0);
            end
        end

        $display("");
        if (errors == 0)
            $display("=== ALL TESTS PASSED ===");
        else
            $display("=== %0d TEST(S) FAILED ===", errors);
        $finish;
    end

endmodule
