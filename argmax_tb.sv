// argmax_tb.sv
//
// Tests argmax against three score arrays:
//   1. Scores from the FC synthetic test: [0,127,0,0,0,0,127,0,127,0]
//      Two classes tie at 127 -- class 1 (first seen) should win.
//   2. Single clear winner: class 7 is highest.
//   3. All zeros: class 0 wins (first seen).

`timescale 1ns / 1ps

module argmax_tb;

    logic       clk, rst, start, done;
    logic [3:0] pred, score_addr;
    logic signed [7:0] score_data;

    // Score buffer (loaded per test)
    logic signed [7:0] score_buf [0:9];
    always_ff @(posedge clk)
        score_data <= score_buf[score_addr];

    argmax dut (
        .clk       (clk), .rst   (rst),
        .start     (start), .done(done), .pred(pred),
        .score_addr(score_addr), .score_data(score_data)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer i, errors, cycle_count;

    task run_argmax(input integer expected_pred, input string label);
        start <= 1; @(posedge clk); #1; start <= 0;
        cycle_count = 0;
        while (!done) begin
            @(posedge clk); #1;
            cycle_count = cycle_count + 1;
            if (cycle_count > 50) begin
                $display("FAIL: timeout on %s", label); errors=errors+1; $finish;
            end
        end
        if (pred === expected_pred)
            $display("PASS: pred=%0d (expected %0d) [%s]", pred, expected_pred, label);
        else begin
            $display("FAIL: pred=%0d (expected %0d) [%s]", pred, expected_pred, label);
            errors = errors + 1;
        end
    endtask

    initial begin
        errors = 0;
        rst <= 1; start <= 0;
        repeat(3) @(posedge clk); #1;
        rst <= 0; @(posedge clk); #1;

        // Test 1: [0,127,0,0,0,0,127,0,127,0] -> class 1 wins (first max)
        $display("--- Test 1: tie at 127 (classes 1,6,8) -> class 1 wins ---");
        score_buf[0]=0; score_buf[1]=127; score_buf[2]=0;
        score_buf[3]=0; score_buf[4]=0;   score_buf[5]=0;
        score_buf[6]=127; score_buf[7]=0; score_buf[8]=127;
        score_buf[9]=0;
        run_argmax(1, "tie->class1");

        // Test 2: clear winner at class 7
        $display("--- Test 2: clear winner at class 7 ---");
        for (i = 0; i < 10; i = i + 1) score_buf[i] = 0;
        score_buf[7] = 100;
        run_argmax(7, "class7 wins");

        // Test 3: all zeros -> class 0
        $display("--- Test 3: all zeros -> class 0 ---");
        for (i = 0; i < 10; i = i + 1) score_buf[i] = 0;
        run_argmax(0, "all zeros->class0");

        // Test 4: last class wins (class 9)
        $display("--- Test 4: class 9 is highest ---");
        for (i = 0; i < 10; i = i + 1) score_buf[i] = i * 10;
        score_buf[9] = 127;
        run_argmax(9, "class9 wins");

        $display("");
        if (errors == 0) $display("=== ALL TESTS PASSED ===");
        else             $display("=== %0d TEST(S) FAILED ===", errors);
        $finish;
    end

endmodule
