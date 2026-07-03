// maxpool_tb.sv
//
// Testbench for maxpool, parameterized as pool1 (4x26x26 -> 4x13x13).
//
// Uses a synthetic input buffer filled with a known pattern so we can
// compute the expected max values independently in Python/by hand:
//   input[ch][row][col] = (ch*26*26 + row*26 + col) % 127
//
// Golden test cases:
//   1. Window (ch=0, py=0, px=0): covers rows 0-1, cols 0-1
//      values: [0,1,26,27] % 127 -> max = 27
//   2. Window (ch=0, py=3, px=3): covers rows 6-7, cols 6-7
//      values at flat addrs: [162,163,188,189] % 127 -> max = 62
//      (with the real MNIST image this window contains 127, but
//       we use the synthetic pattern here for a deterministic test)
//   3. All 676 output pixels written (coverage)
//   4. No negative values in output (all inputs are non-negative)

`timescale 1ns / 1ps

module maxpool_tb;

    // Instantiate as pool1 (4x26x26 -> 4x13x13)
    localparam IN_CH   = 4;
    localparam IN_H    = 26;
    localparam IN_W    = 26;
    localparam IN_SIZE = IN_CH * IN_H * IN_W;   // 2704
    localparam OUT_H   = 13;
    localparam OUT_W   = 13;
    localparam OUT_SIZE = IN_CH * OUT_H * OUT_W; // 676

    logic        clk, rst, start, done;
    logic [11:0] in_addr;
    logic signed [7:0] in_data;
    logic        out_wen;
    logic [9:0]  out_addr;
    logic signed [7:0] out_data;

    // Synthetic input memory: value = flat_index % 127
    logic signed [7:0] in_mem [0:IN_SIZE-1];

    // Registered read (matches input_buf's synchronous read behavior)
    always_ff @(posedge clk)
        in_data <= in_mem[in_addr];

    // Instantiate pool1
    maxpool #(
        .IN_CH  (IN_CH),
        .IN_H   (IN_H),
        .IN_W   (IN_W)
    ) dut (
        .clk    (clk),  .rst    (rst),
        .start  (start),.done   (done),
        .in_addr(in_addr), .in_data(in_data),
        .out_wen(out_wen), .out_addr(out_addr), .out_data(out_data)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // Capture all output writes
    logic signed [7:0] output_buf [0:OUT_SIZE-1];
    always_ff @(posedge clk)
        if (out_wen) output_buf[out_addr] <= out_data;

    integer i, errors, cycle_count;

    initial begin
        errors = 0;
        cycle_count = 0;

        // Fill synthetic input
        for (i = 0; i < IN_SIZE; i = i + 1)
            in_mem[i] = i % 127;

        // Initialize output sentinel
        for (i = 0; i < OUT_SIZE; i = i + 1)
            output_buf[i] = -1;

        // Reset
        rst <= 1; start <= 0;
        repeat(3) @(posedge clk); #1;
        rst <= 0; @(posedge clk); #1;

        // Start pooling
        $display("Starting maxpool test (pool1: 4x26x26 -> 4x13x13)...");
        start <= 1; @(posedge clk); #1; start <= 0;

        // Wait for done
        while (!done) begin
            @(posedge clk); #1;
            cycle_count = cycle_count + 1;
            if (cycle_count > 20000) begin
                $display("FAIL: timeout"); errors = errors + 1; $finish;
            end
        end
        $display("done after %0d cycles", cycle_count);

        // Test 1: golden window (ch=0, py=0, px=0)
        // Input flat indices: 0,1,26,27 -> values 0,1,26,27 -> max=27
        $display("");
        $display("--- Test 1: window (ch=0, py=0, px=0), expect max=27 ---");
        begin
            integer exp_max;
            exp_max = 27;  // max(0%127, 1%127, 26%127, 27%127) = 27
            if (output_buf[0] === exp_max)
                $display("PASS: output_buf[0] = %0d", output_buf[0]);
            else begin
                $display("FAIL: output_buf[0] = %0d, expected %0d",
                         output_buf[0], exp_max);
                errors = errors + 1;
            end
        end

        // Test 2: golden window (ch=0, py=3, px=3)
        // Covers input rows 6-7, cols 6-7
        // Flat addrs: 6*26+6=162, 6*26+7=163, 7*26+6=188, 7*26+7=189
        // Values: 162%127=35, 163%127=36, 188%127=61, 189%127=62
        // Max = 62; output addr = 0*169 + 3*13 + 3 = 42
        $display("");
        $display("--- Test 2: window (ch=0, py=3, px=3), expect max=62 ---");
        begin
            integer v0, v1, v2, v3, exp_max2;
            v0 = 162 % 127; v1 = 163 % 127;
            v2 = 188 % 127; v3 = 189 % 127;
            exp_max2 = v3;  // 62 is largest
            $display("    Input values: %0d, %0d, %0d, %0d -> max=%0d",
                     v0, v1, v2, v3, exp_max2);
            if (output_buf[42] === exp_max2)
                $display("PASS: output_buf[42] = %0d", output_buf[42]);
            else begin
                $display("FAIL: output_buf[42] = %0d, expected %0d",
                         output_buf[42], exp_max2);
                errors = errors + 1;
            end
        end

        // Test 3: all 676 output pixels written
        $display("");
        $display("--- Test 3: all %0d outputs written ---", OUT_SIZE);
        begin
            integer unwritten;
            unwritten = 0;
            for (i = 0; i < OUT_SIZE; i = i + 1)
                if (output_buf[i] === -1) unwritten = unwritten + 1;
            if (unwritten == 0)
                $display("PASS: all %0d pixels written", OUT_SIZE);
            else begin
                $display("FAIL: %0d unwritten", unwritten);
                errors = errors + 1;
            end
        end

        // Test 4: no negative outputs (all inputs >= 0)
        $display("");
        $display("--- Test 4: no negative outputs ---");
        begin
            integer neg_count;
            neg_count = 0;
            for (i = 0; i < OUT_SIZE; i = i + 1)
                if (output_buf[i] < 0) neg_count = neg_count + 1;
            if (neg_count == 0)
                $display("PASS: all values >= 0");
            else begin
                $display("FAIL: %0d negative values", neg_count);
                errors = errors + 1;
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
