// mac_unit_tb.sv
//
// Testbench for mac_unit. Feeds in the exact 9 input/weight pairs from
// verify_mac.py's verified output and checks the final accumulator
// equals 48387 -- the same number Python computed by hand.
//
// Golden test case (from verify_mac.py, MNIST test image #0, channel 0,
// row=7, col=6):
//   inputs:  42  92  79 111 127 127  33  57  36
//   weights: -16 54 101  59  85 126  11  39   5
//   expected final accumulator: 48387
//
// IMPORTANT testbench-writing lesson baked into this file: all signals
// that drive the DUT (rst, clear, valid, in_val, weight) are assigned
// with nonblocking assignments (<=), not blocking (=). If you use
// blocking assignments here, you can hit a real race condition: the
// DUT's always_ff block and this testbench's stimulus-driving process
// are technically two separate processes both triggered by the same
// posedge clk edge, and the simulator does not guarantee which one
// runs first. With blocking assignments, it's possible for the DUT to
// sample an old/stale value of a signal you thought you'd already
// updated before that edge. Nonblocking assignments schedule the
// update to happen in a separate "NBA" update region after all
// processes have evaluated for that time step, which removes the
// ambiguity entirely. This is exactly the kind of bug that's painful
// to debug on real hardware, so it's worth internalizing here in
// simulation first.

`timescale 1ns / 1ps

module mac_unit_tb;

    logic               clk;
    logic               rst;
    logic               clear;
    logic               valid;
    logic signed [7:0]  in_val;
    logic signed [7:0]  weight;
    logic signed [19:0] acc;

    // Instantiate the unit under test
    mac_unit uut (
        .clk    (clk),
        .rst    (rst),
        .clear  (clear),
        .valid  (valid),
        .in_val (in_val),
        .weight (weight),
        .acc    (acc)
    );

    // Clock generator: 10ns period (100MHz), toggles forever
    initial clk = 0;
    always #5 clk = ~clk;

    // The 9 golden input/weight pairs, in the same order as the
    // Python trace (ky=0..2, kx=0..2, row-major).
    logic signed [7:0] inputs [0:8];
    logic signed [7:0] weights[0:8];

    // Expected running accumulator after each of the 9 steps, taken
    // directly from the "running_acc" column of verify_mac.py's output.
    // Checking every intermediate step (not just the final value)
    // means a bug shows up immediately at the step that's wrong,
    // instead of only producing a mismatched final answer with no clue
    // where the error was introduced.
    integer expected_trace[0:8];

    integer i;
    integer errors;

    initial begin
        // Initialize golden test vectors first, before anything else runs.
        inputs[0] = 42;   inputs[1] = 92;   inputs[2] = 79;
        inputs[3] = 111;  inputs[4] = 127;  inputs[5] = 127;
        inputs[6] = 33;   inputs[7] = 57;   inputs[8] = 36;

        weights[0] = -16; weights[1] = 54;  weights[2] = 101;
        weights[3] = 59;  weights[4] = 85;  weights[5] = 126;
        weights[6] = 11;  weights[7] = 39;  weights[8] = 5;

        expected_trace[0] = -672;  expected_trace[1] = 4296;  expected_trace[2] = 12275;
        expected_trace[3] = 18824; expected_trace[4] = 29619; expected_trace[5] = 45621;
        expected_trace[6] = 45984; expected_trace[7] = 48207; expected_trace[8] = 48387;

        errors = 0;

        // Reset the accumulator before starting. Nonblocking assignments
        // throughout, per the note above.
        rst    <= 1;
        clear  <= 0;
        valid  <= 0;
        in_val <= 0;
        weight <= 0;
        @(posedge clk);
        #1; // let the reset take effect and settle before changing rst
        rst <= 0;
        @(posedge clk);
        #1;

        $display("Starting MAC unit test -- feeding 9 input/weight pairs...");
        $display("%4s %8s %8s %12s %12s", "step", "in_val", "weight", "acc", "expected");

        // Feed one pair per clock cycle
        for (i = 0; i < 9; i = i + 1) begin
            in_val <= inputs[i];
            weight <= weights[i];
            valid  <= 1;
            @(posedge clk);
            #1; // settling delay so acc reflects this edge's update before we read it
            $display("%4d %8d %8d %12d %12d", i, in_val, weight, acc, expected_trace[i]);
            if (acc !== expected_trace[i]) begin
                $display("  MISMATCH at step %0d: got acc=%0d, expected %0d", i, acc, expected_trace[i]);
                errors = errors + 1;
            end
        end

        valid <= 0;

        $display("");
        if (errors == 0) begin
            $display("PASS: all 9 steps matched expected values. Final acc = %0d (expected 48387)", acc);
        end else begin
            $display("FAIL: %0d step(s) mismatched. See MISMATCH lines above.", errors);
        end

        // Bonus check: clear should reset acc to 0 on the next cycle
        clear <= 1;
        @(posedge clk);
        #1;
        clear <= 0;
        if (acc === 20'sd0) begin
            $display("PASS: clear correctly reset acc to 0.");
        end else begin
            $display("FAIL: clear did not reset acc (acc=%0d).", acc);
            errors = errors + 1;
        end

        $display("");
        if (errors == 0) begin
            $display("=== ALL TESTS PASSED ===");
        end else begin
            $display("=== %0d TEST(S) FAILED ===", errors);
        end

        $finish;
    end

endmodule
