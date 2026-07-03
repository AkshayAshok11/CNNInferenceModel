// conv1_tb.sv
//
// Testbench for the full conv1 layer. Runs a complete inference on
// MNIST test image #0 (loaded from input_image.hex) and verifies:
//
//   1. The done signal fires exactly once after all 2704 pixels
//   2. The output at (oc=0, oy=7, ox=6) equals 127 -- the golden
//      value computed from the verified Python accumulator of 48387
//      plus bias of -2, clipped to int8 by ReLU+saturation
//   3. The output write address for that pixel is 188
//      (oc*676 + oy*26 + ox = 0*676 + 7*26 + 6)
//   4. No output pixel exceeds the int8 range [0,127] after ReLU
//
// The testbench also stores all 2704 output values in a local array
// and prints a summary at the end showing how many pixels per channel
// are nonzero (a quick sanity check that the layer is doing real work,
// not just outputting zeros).

`timescale 1ns / 1ps

module conv1_tb;

    logic        clk, rst, start, done;
    logic        out_wen;
    logic [11:0] out_addr;
    logic signed [7:0] out_data;

    conv1 dut (
        .clk     (clk),
        .rst     (rst),
        .start   (start),
        .done    (done),
        .out_wen (out_wen),
        .out_addr(out_addr),
        .out_data(out_data)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // Local output buffer: capture every write from conv1
    logic signed [7:0] output_buf [0:2703];

    integer errors;
    integer cycle_count;
    integer done_count;
    integer i, ch;
    integer nonzero_count;

    // Monitor: capture every output write as it happens
    always_ff @(posedge clk) begin
        if (out_wen) begin
            output_buf[out_addr] <= out_data;
        end
    end

    initial begin
        errors     = 0;
        cycle_count = 0;
        done_count  = 0;
        start <= 0;

        // Initialize output buffer to a sentinel value so we can
        // detect addresses that were never written
        for (i = 0; i < 2704; i = i + 1)
            output_buf[i] = -128;  // -128 = sentinel, conv1 never outputs this

        // Reset
        rst <= 1;
        repeat (3) @(posedge clk);
        #1;
        rst <= 0;
        @(posedge clk); #1;

        // Fire start
        $display("Starting conv1 inference on MNIST test image #0...");
        start <= 1;
        @(posedge clk); #1;
        start <= 0;

        // Wait for done, counting cycles
        while (!done) begin
            @(posedge clk); #1;
            cycle_count = cycle_count + 1;
            if (cycle_count > 40000) begin
                $display("FAIL: done never fired -- timeout at %0d cycles", cycle_count);
                errors = errors + 1;
                $finish;
            end
        end
        done_count = done_count + 1;

        $display("done fired after %0d cycles (expected ~29744)", cycle_count);
        $display("");

        // Test 1: done fired exactly once (not repeatedly)
        // (wait a few more cycles to confirm it doesn't re-fire)
        repeat (5) begin
            @(posedge clk); #1;
            if (done) done_count = done_count + 1;
        end
        if (done_count == 1)
            $display("PASS: done fired exactly once");
        else begin
            $display("FAIL: done fired %0d times (expected 1)", done_count);
            errors = errors + 1;
        end

        // Test 2: golden pixel check
        // oc=0, oy=7, ox=6 -> addr=188, expected value=127
        $display("");
        $display("--- Golden pixel check (oc=0, oy=7, ox=6, addr=188) ---");
        $display("    Python: acc=48387, bias=-2, biased=48385 -> ReLU/clip -> 127");
        if (output_buf[188] === 8'sd127) begin
            $display("PASS: output_buf[188] = %0d (expected 127)", output_buf[188]);
        end else begin
            $display("FAIL: output_buf[188] = %0d (expected 127)", output_buf[188]);
            errors = errors + 1;
        end

        // Test 3: all 2704 addresses were written (no sentinels remain)
        $display("");
        $display("--- Coverage check: all 2704 addresses written ---");
        begin
            integer unwritten;
            unwritten = 0;
            for (i = 0; i < 2704; i = i + 1) begin
                if (output_buf[i] === -128) unwritten = unwritten + 1;
            end
            if (unwritten == 0)
                $display("PASS: all 2704 output pixels written");
            else begin
                $display("FAIL: %0d addresses never written", unwritten);
                errors = errors + 1;
            end
        end

        // Test 4: all values in [0,127] after ReLU (no negatives)
        $display("");
        $display("--- Range check: all values in [0,127] ---");
        begin
            integer range_errors;
            range_errors = 0;
            for (i = 0; i < 2704; i = i + 1) begin
                if (output_buf[i] < 0) begin
                    if (range_errors < 5)  // only print first 5
                        $display("  FAIL addr=%0d: value=%0d is negative (ReLU should prevent this)",
                                 i, output_buf[i]);
                    range_errors = range_errors + 1;
                end
            end
            if (range_errors == 0)
                $display("PASS: all values >= 0 (ReLU working correctly)");
            else begin
                $display("FAIL: %0d negative values found", range_errors);
                errors = errors + 1;
            end
        end

        // Summary: nonzero pixels per channel
        // (confirms the layer is producing real feature maps, not all zeros)
        $display("");
        $display("--- Nonzero pixel count per output channel ---");
        for (ch = 0; ch < 4; ch = ch + 1) begin
            nonzero_count = 0;
            for (i = 0; i < 676; i = i + 1) begin
                if (output_buf[ch * 676 + i] !== 8'sd0)
                    nonzero_count = nonzero_count + 1;
            end
            $display("  Channel %0d: %0d / 676 pixels nonzero (%.1f%%)",
                     ch, nonzero_count, nonzero_count * 100.0 / 676.0);
        end

        $display("");
        if (errors == 0)
            $display("=== ALL TESTS PASSED ===");
        else
            $display("=== %0d TEST(S) FAILED ===", errors);

        $finish;
    end

endmodule
