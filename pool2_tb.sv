// pool2_tb.sv
//
// Testbench for pool2: maxpool parameterized as 8x11x11 -> 8x5x5.
// Uses synthetic input: value = flat_index % 127
//
// Tests:
//   1. Window (ch=0, py=0, px=0): pixels [0,1,11,12], max=12, addr=0
//   2. Window (ch=3, py=2, px=4): pixels [34,35,45,46], max=46, addr=89
//   3. All 200 output pixels written
//   4. No negative values

`timescale 1ns / 1ps

module pool2_tb;

    localparam IN_CH   = 8;
    localparam IN_H    = 11;
    localparam IN_W    = 11;
    localparam IN_SIZE = IN_CH * IN_H * IN_W;   // 968
    localparam OUT_H   = 5;
    localparam OUT_W   = 5;
    localparam OUT_SIZE = IN_CH * OUT_H * OUT_W; // 200

    logic        clk, rst, start, done;
    logic [9:0]  in_addr;   // 10 bits covers 0..967
    logic signed [7:0] in_data;
    logic        out_wen;
    logic [7:0]  out_addr;  // 8 bits covers 0..199
    logic signed [7:0] out_data;

    // Synthetic conv2 output buffer: value = flat_index % 127
    logic signed [7:0] in_mem [0:IN_SIZE-1];
    always_ff @(posedge clk)
        in_data <= in_mem[in_addr];

    // Instantiate as pool2
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

    logic signed [7:0] output_buf [0:OUT_SIZE-1];
    always_ff @(posedge clk)
        if (out_wen) output_buf[out_addr] <= out_data;

    integer i, errors, cycle_count;

    initial begin
        errors = 0; cycle_count = 0;

        for (i = 0; i < IN_SIZE; i = i + 1)
            in_mem[i] = i % 127;
        for (i = 0; i < OUT_SIZE; i = i + 1)
            output_buf[i] = -1;  // sentinel

        rst <= 1; start <= 0;
        repeat(3) @(posedge clk); #1;
        rst <= 0; @(posedge clk); #1;

        $display("Starting pool2 test (8x11x11 -> 8x5x5)...");
        start <= 1; @(posedge clk); #1; start <= 0;

        while (!done) begin
            @(posedge clk); #1;
            cycle_count = cycle_count + 1;
            if (cycle_count > 10000) begin
                $display("FAIL: timeout"); errors = errors + 1; $finish;
            end
        end
        @(posedge clk); #1;  // settle final write
        $display("done after %0d cycles", cycle_count);

        // Test 1
        $display("\n--- Test 1: ch=0 py=0 px=0, expect max=12 at addr=0 ---");
        if (output_buf[0] === 8'sd12)
            $display("PASS: output_buf[0] = 12");
        else begin
            $display("FAIL: output_buf[0] = %0d (expected 12)", output_buf[0]);
            errors = errors + 1;
        end

        // Test 2
        $display("\n--- Test 2: ch=3 py=2 px=4, expect max=46 at addr=89 ---");
        if (output_buf[89] === 8'sd46)
            $display("PASS: output_buf[89] = 46");
        else begin
            $display("FAIL: output_buf[89] = %0d (expected 46)", output_buf[89]);
            errors = errors + 1;
        end

        // Test 3
        $display("\n--- Test 3: all 200 outputs written ---");
        begin
            integer unwritten;
            unwritten = 0;
            for (i = 0; i < OUT_SIZE; i = i + 1)
                if (output_buf[i] === -1) unwritten = unwritten + 1;
            if (unwritten == 0)
                $display("PASS: all 200 pixels written");
            else begin
                $display("FAIL: %0d unwritten", unwritten);
                errors = errors + 1;
            end
        end

        // Test 4
        $display("\n--- Test 4: no negative values ---");
        begin
            integer neg;
            neg = 0;
            for (i = 0; i < OUT_SIZE; i = i + 1)
                if (output_buf[i] < 0) neg = neg + 1;
            if (neg == 0)
                $display("PASS: all values >= 0");
            else begin
                $display("FAIL: %0d negative values", neg);
                errors = errors + 1;
            end
        end

        $display("");
        if (errors == 0) $display("=== ALL TESTS PASSED ===");
        else             $display("=== %0d TEST(S) FAILED ===", errors);
        $finish;
    end

endmodule
