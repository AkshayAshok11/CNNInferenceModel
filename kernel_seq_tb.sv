// kernel_seq_tb.sv
//
// Testbench for kernel_seq. Verifies:
//   1. ky/kx sequence is correct (0,0),(0,1),(0,2),(1,0),...,(2,2)
//   2. mac_clear fires exactly once before mac_valid goes high
//   3. mac_valid is high for exactly 9 cycles
//   4. done fires exactly once, on the 9th cycle (step 8)
//   5. Back-to-back starts work correctly (second pixel starts
//      immediately after first completes)
//   6. Idle stays idle without a start pulse

`timescale 1ns / 1ps

module kernel_seq_tb;

    logic       clk, rst, start;
    logic [1:0] ky, kx;
    logic       mac_valid, mac_clear, done;

    kernel_seq uut (
        .clk      (clk),
        .rst      (rst),
        .start    (start),
        .ky       (ky),
        .kx       (kx),
        .mac_valid(mac_valid),
        .mac_clear(mac_clear),
        .done     (done)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // Expected ky/kx sequence for a 3x3 kernel, row-major
    logic [1:0] exp_ky [0:8];
    logic [1:0] exp_kx [0:8];

    integer i;
    integer errors;

    // Helper task: wait one clock, then read outputs after settling
    task tick;
        @(posedge clk);
        #1;
    endtask

    // Helper task: run one full 9-step MAC sequence from a start pulse
    // and verify every signal at every step.
    task run_pixel(input integer pixel_num);
        // Pulse start for one cycle
        start <= 1;
        tick;
        start <= 0;

        // Expect CLEARING state this cycle: clear high, valid low
        if (mac_clear !== 1'b1) begin
            $display("PIXEL %0d FAIL: expected mac_clear=1 on cycle after start, got %b", pixel_num, mac_clear);
            errors = errors + 1;
        end
        if (mac_valid !== 1'b0) begin
            $display("PIXEL %0d FAIL: expected mac_valid=0 during clear cycle, got %b", pixel_num, mac_valid);
            errors = errors + 1;
        end
        if (done !== 1'b0) begin
            $display("PIXEL %0d FAIL: done should not be high during clear cycle", pixel_num);
            errors = errors + 1;
        end
        tick;

        // Now 9 RUNNING cycles
        for (i = 0; i < 9; i = i + 1) begin
            if (mac_valid !== 1'b1) begin
                $display("PIXEL %0d FAIL step %0d: mac_valid should be 1, got %b", pixel_num, i, mac_valid);
                errors = errors + 1;
            end
            if (mac_clear !== 1'b0) begin
                $display("PIXEL %0d FAIL step %0d: mac_clear should be 0 during running, got %b", pixel_num, i, mac_clear);
                errors = errors + 1;
            end
            if (ky !== exp_ky[i] || kx !== exp_kx[i]) begin
                $display("PIXEL %0d FAIL step %0d: expected ky=%0d kx=%0d, got ky=%0d kx=%0d",
                         pixel_num, i, exp_ky[i], exp_kx[i], ky, kx);
                errors = errors + 1;
            end
            // done should only be high on step 8
            if (i == 8 && done !== 1'b1) begin
                $display("PIXEL %0d FAIL: done should be high on step 8, got %b", pixel_num, done);
                errors = errors + 1;
            end
            if (i != 8 && done !== 1'b0) begin
                $display("PIXEL %0d FAIL step %0d: done should be low until step 8, got %b", pixel_num, i, done);
                errors = errors + 1;
            end
            $display("  pixel=%0d step=%0d ky=%0d kx=%0d valid=%b clear=%b done=%b",
                     pixel_num, i, ky, kx, mac_valid, mac_clear, done);
            if (i < 8) tick;
        end
        // After the 9th step display, advance one more clock and check idle
        tick;
        if (mac_valid !== 1'b0) begin
            $display("PIXEL %0d FAIL: mac_valid should drop after step 8, got %b", pixel_num, mac_valid);
            errors = errors + 1;
        end
        if (done !== 1'b0) begin
            $display("PIXEL %0d FAIL: done should drop after step 8, got %b", pixel_num, done);
            errors = errors + 1;
        end
    endtask

    initial begin
        // Set up expected ky/kx sequence
        exp_ky[0]=0; exp_kx[0]=0;
        exp_ky[1]=0; exp_kx[1]=1;
        exp_ky[2]=0; exp_kx[2]=2;
        exp_ky[3]=1; exp_kx[3]=0;
        exp_ky[4]=1; exp_kx[4]=1;
        exp_ky[5]=1; exp_kx[5]=2;
        exp_ky[6]=2; exp_kx[6]=0;
        exp_ky[7]=2; exp_kx[7]=1;
        exp_ky[8]=2; exp_kx[8]=2;

        errors = 0;
        start <= 0;

        // Reset
        rst <= 1;
        tick;
        tick;
        rst <= 0;
        tick;

        // Test 1: verify it stays idle without a start pulse
        $display("--- Test 1: idle stays idle ---");
        repeat (3) tick;
        if (mac_valid !== 0 || mac_clear !== 0 || done !== 0) begin
            $display("FAIL: signals should all be 0 in idle");
            errors = errors + 1;
        end else begin
            $display("PASS: idle stays idle");
        end

        // Test 2: first pixel - verify full 9-step sequence
        $display("--- Test 2: first pixel, full 9-step sequence ---");
        run_pixel(0);

        // Test 3: second pixel immediately after first (back-to-back)
        $display("--- Test 3: second pixel, back-to-back start ---");
        run_pixel(1);

        // Test 4: wait a few cycles then start again (non-immediate restart)
        $display("--- Test 4: delayed restart ---");
        repeat (4) tick;
        run_pixel(2);

        $display("");
        if (errors == 0)
            $display("=== ALL TESTS PASSED ===");
        else
            $display("=== %0d TEST(S) FAILED ===", errors);

        $finish;
    end

endmodule
