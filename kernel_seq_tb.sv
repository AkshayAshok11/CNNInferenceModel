// kernel_seq_tb.sv (updated for DRAINING state)
//
// done now fires ONE cycle after the last mac_valid (in DRAINING state),
// giving registered memory reads time to deliver the final pixel.
//
// New timing per pixel:
//   cycle 1:    CLEARING  (mac_clear=1, valid=0, done=0)
//   cycles 2-10: RUNNING  (mac_valid=1, done=0)
//   cycle 11:   DRAINING  (mac_valid=0, done=1)
//   cycle 12+:  IDLE

`timescale 1ns / 1ps
module kernel_seq_tb;
    logic       clk, rst, start;
    logic [1:0] ky, kx;
    logic       mac_valid, mac_clear, done;

    kernel_seq uut(.clk(clk),.rst(rst),.start(start),
                   .ky(ky),.kx(kx),.mac_valid(mac_valid),
                   .mac_clear(mac_clear),.done(done));

    initial clk = 0;
    always #5 clk = ~clk;

    logic [1:0] exp_ky[0:8];
    logic [1:0] exp_kx[0:8];
    integer i, errors;

    task tick; @(posedge clk); #1; endtask

    task run_pixel(input integer pixel_num);
        // Pulse start
        start <= 1; tick; start <= 0;

        // Cycle 1: CLEARING
        if (mac_clear !== 1'b1) begin
            $display("PIXEL %0d FAIL: expected clear=1 after start, got %b", pixel_num, mac_clear);
            errors = errors + 1;
        end
        if (mac_valid !== 1'b0) begin
            $display("PIXEL %0d FAIL: valid should be 0 during clear", pixel_num);
            errors = errors + 1;
        end
        tick;

        // Cycles 2-10: RUNNING (9 valid cycles)
        for (i = 0; i < 9; i = i + 1) begin
            if (mac_valid !== 1'b1) begin
                $display("PIXEL %0d FAIL step %0d: valid should be 1", pixel_num, i);
                errors = errors + 1;
            end
            if (ky !== exp_ky[i] || kx !== exp_kx[i]) begin
                $display("PIXEL %0d FAIL step %0d: expected ky=%0d kx=%0d, got %0d %0d",
                         pixel_num, i, exp_ky[i], exp_kx[i], ky, kx);
                errors = errors + 1;
            end
            if (done !== 1'b0) begin
                $display("PIXEL %0d FAIL step %0d: done should be 0 during valid", pixel_num, i);
                errors = errors + 1;
            end
            $display("  pixel=%0d step=%0d ky=%0d kx=%0d valid=%b clear=%b done=%b",
                     pixel_num, i, ky, kx, mac_valid, mac_clear, done);
            tick;
        end

        // Cycle 11: DRAINING (done=1, valid=0)
        if (done !== 1'b1) begin
            $display("PIXEL %0d FAIL: done should be 1 in DRAINING, got %b", pixel_num, done);
            errors = errors + 1;
        end
        if (mac_valid !== 1'b0) begin
            $display("PIXEL %0d FAIL: valid should be 0 in DRAINING, got %b", pixel_num, mac_valid);
            errors = errors + 1;
        end
        $display("  pixel=%0d DRAINING done=%b valid=%b", pixel_num, done, mac_valid);
        tick;

        // Cycle 12: IDLE
        if (done !== 1'b0 || mac_valid !== 1'b0) begin
            $display("PIXEL %0d FAIL: should be IDLE after DRAINING", pixel_num);
            errors = errors + 1;
        end
    endtask

    initial begin
        exp_ky[0]=0; exp_kx[0]=0; exp_ky[1]=0; exp_kx[1]=1; exp_ky[2]=0; exp_kx[2]=2;
        exp_ky[3]=1; exp_kx[3]=0; exp_ky[4]=1; exp_kx[4]=1; exp_ky[5]=1; exp_kx[5]=2;
        exp_ky[6]=2; exp_kx[6]=0; exp_ky[7]=2; exp_kx[7]=1; exp_ky[8]=2; exp_kx[8]=2;
        errors = 0; start <= 0;

        rst <= 1; repeat(3) @(posedge clk); #1; rst <= 0; @(posedge clk); #1;

        $display("--- Test 1: idle ---");
        repeat(3) tick;
        if (mac_valid!==0 || mac_clear!==0 || done!==0) begin
            $display("FAIL: idle should have all signals 0"); errors=errors+1;
        end else $display("PASS: idle stays idle");

        $display("--- Test 2: first pixel ---");
        run_pixel(0);

        $display("--- Test 3: back-to-back ---");
        run_pixel(1);

        $display("--- Test 4: delayed restart ---");
        repeat(4) tick;
        run_pixel(2);

        $display("");
        if (errors == 0) $display("=== ALL TESTS PASSED ===");
        else             $display("=== %0d TEST(S) FAILED ===", errors);
        $finish;
    end
endmodule