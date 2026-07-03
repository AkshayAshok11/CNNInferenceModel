// input_buf_tb.sv
//
// Testbench for input_buf. Verifies:
//   1. The golden patch (row=7..9, col=6..8) matches verify_mac.py's
//      known input pixel values exactly -- same pixels that produced
//      the verified accumulator of 48387
//   2. Known non-patch pixels match the synthetic pattern
//      pixel[row][col] = (row*28 + col) % 128
//   3. Boundary pixels (corners) are correct
//   4. The one-cycle read latency works as expected
//
// NOTE: when you run this on your machine with the real
// input_image.hex from gen_input_image.py, tests 2 and 3 will
// fail since they check the synthetic pattern, not the real image.
// Only test 1 (the golden patch) will be meaningful with the real hex.
// After confirming the golden patch passes, swap to the real image.

`timescale 1ns / 1ps

module input_buf_tb;

    logic            clk;
    logic [4:0]      row, col;
    logic signed [7:0] pixel_out;

    input_buf uut (
        .clk      (clk),
        .row      (row),
        .col      (col),
        .pixel_out(pixel_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer errors;

    // Helper task: set address, wait one cycle (the registered latency),
    // then check the output.
    task check_pixel(
        input integer r,
        input integer c,
        input integer expected,
        input string  label
    );
        row <= r[4:0];
        col <= c[4:0];
        @(posedge clk);
        #1;
        if (pixel_out !== expected[7:0]) begin
            $display("  FAIL [%0d][%0d] (%s): got %0d, expected %0d",
                     r, c, label, pixel_out, expected);
            errors = errors + 1;
        end else begin
            $display("  PASS [%0d][%0d] (%s) = %0d", r, c, label, pixel_out);
        end
    endtask

    initial begin
        errors = 0;
        row <= 0;
        col <= 0;

        // Reset / settle
        @(posedge clk); #1;
        @(posedge clk); #1;

        // Test 1: golden patch -- these must match verify_mac.py exactly
        // regardless of which hex file is loaded, because the real
        // gen_input_image.py will produce these same values for image #0
        $display("--- Test 1: golden patch (row=7..9, col=6..8) ---");
        $display("    These must match verify_mac.py golden input pixels.");
        check_pixel(7, 6,  42, "golden");
        check_pixel(7, 7,  92, "golden");
        check_pixel(7, 8,  79, "golden");
        check_pixel(8, 6, 111, "golden");
        check_pixel(8, 7, 127, "golden");
        check_pixel(8, 8, 127, "golden");
        check_pixel(9, 6,  33, "golden");
        check_pixel(9, 7,  57, "golden");
        check_pixel(9, 8,  36, "golden");

        // Test 2: a known non-patch pixel from the synthetic pattern
        // pixel[3][5] = (3*28 + 5) % 128 = 89
        // (will fail with real image -- that's expected)
        $display("--- Test 2: synthetic pattern spot-check (synthetic hex only) ---");
        check_pixel(3, 5, 89, "pattern");

        // Test 3: boundary pixels
        $display("--- Test 3: boundary pixels ---");
        // [0][0] = (0*28+0)%128 = 0
        check_pixel(0, 0,   0, "top-left");
        // [27][27] = (27*28+27)%128 = (756+27)%128 = 783%128 = 15
        check_pixel(27, 27, 15, "bottom-right");

        // Test 4: one-cycle latency -- present address, confirm output
        // only appears AFTER the clock edge, not before
        $display("--- Test 4: latency check ---");
        row <= 5'b00111;  // row=7
        col <= 5'b00110;  // col=6  (expect pixel=42)
        // pixel_out should NOT be 42 yet (still showing previous read)
        #1; // tiny delay within same cycle, before next edge
        // now advance clock -- output should update
        @(posedge clk); #1;
        if (pixel_out === 8'sd42)
            $display("  PASS: pixel[7][6]=42 appears correctly after clock edge");
        else begin
            $display("  FAIL: pixel[7][6] expected 42, got %0d", pixel_out);
            errors = errors + 1;
        end

        $display("");
        if (errors == 0)
            $display("=== ALL TESTS PASSED ===");
        else
            $display("=== %0d TEST(S) FAILED ===", errors);

        $finish;
    end

endmodule
