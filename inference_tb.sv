// inference_tb.sv -- full pipeline inference testbench
`timescale 1ns / 1ps

module inference_tb;

    logic clk, rst;
    initial clk = 0;
    always #5 clk = ~clk;

    // ---------------------------------------------------------------
    // Intermediate buffers
    // ---------------------------------------------------------------
    logic signed [7:0] conv1_buf [0:2703];  // 4x26x26
    logic signed [7:0] pool1_buf [0:675];   // 4x13x13
    logic signed [7:0] conv2_buf [0:967];   // 8x11x11
    logic signed [7:0] pool2_buf [0:199];   // 8x5x5
    logic signed [7:0] score_buf [0:9];     // 10 class scores

    // ---------------------------------------------------------------
    // Synchronous reads from each intermediate buffer
    // ---------------------------------------------------------------
    logic [11:0] pool1_in_addr;  logic signed [7:0] pool1_in_data;
    logic [9:0]  conv2_in_addr;  logic signed [7:0] conv2_in_data;
    logic [9:0]  pool2_in_addr;  logic signed [7:0] pool2_in_data;
    logic [7:0]  fc_in_addr;     logic signed [7:0] fc_in_data;
    logic [3:0]  argmax_addr;    logic signed [7:0] argmax_data;

    always_ff @(posedge clk) pool1_in_data <= conv1_buf[pool1_in_addr];
    always_ff @(posedge clk) conv2_in_data <= pool1_buf[conv2_in_addr];
    always_ff @(posedge clk) pool2_in_data <= conv2_buf[pool2_in_addr];
    always_ff @(posedge clk) fc_in_data    <= pool2_buf[fc_in_addr];
    always_ff @(posedge clk) argmax_data   <= score_buf[argmax_addr];

    // ---------------------------------------------------------------
    // Layer control
    // ---------------------------------------------------------------
    logic conv1_start, conv1_done;
    logic pool1_start, pool1_done;
    logic conv2_start, conv2_done;
    logic pool2_start, pool2_done;
    logic fc_start,    fc_done;
    logic argmax_start,argmax_done;
    logic [3:0] pred;

    // Output write ports -> capture into buffers
    logic        conv1_wen; logic [11:0] conv1_waddr; logic signed [7:0] conv1_wdata;
    logic        pool1_wen; logic [9:0]  pool1_waddr; logic signed [7:0] pool1_wdata;
    logic        conv2_wen; logic [9:0]  conv2_waddr; logic signed [7:0] conv2_wdata;
    logic        pool2_wen; logic [7:0]  pool2_waddr; logic signed [7:0] pool2_wdata;
    logic        fc_wen;    logic [3:0]  fc_waddr;    logic signed [7:0] fc_wdata;

    always_ff @(posedge clk) if (conv1_wen) conv1_buf[conv1_waddr] <= conv1_wdata;
    always_ff @(posedge clk) if (pool1_wen) pool1_buf[pool1_waddr] <= pool1_wdata;
    always_ff @(posedge clk) if (conv2_wen) conv2_buf[conv2_waddr] <= conv2_wdata;
    always_ff @(posedge clk) if (pool2_wen) pool2_buf[pool2_waddr] <= pool2_wdata;
    always_ff @(posedge clk) if (fc_wen)    score_buf[fc_waddr]    <= fc_wdata;

    // ---------------------------------------------------------------
    // Instantiations
    // ---------------------------------------------------------------
    conv1 conv1_inst (
        .clk(clk),.rst(rst),.start(conv1_start),.done(conv1_done),
        .out_wen(conv1_wen),.out_addr(conv1_waddr),.out_data(conv1_wdata));

    maxpool #(.IN_CH(4),.IN_H(26),.IN_W(26)) pool1_inst (
        .clk(clk),.rst(rst),.start(pool1_start),.done(pool1_done),
        .in_addr(pool1_in_addr),.in_data(pool1_in_data),
        .out_wen(pool1_wen),.out_addr(pool1_waddr),.out_data(pool1_wdata));

    conv2 conv2_inst (
        .clk(clk),.rst(rst),.start(conv2_start),.done(conv2_done),
        .in_addr(conv2_in_addr),.in_data(conv2_in_data),
        .out_wen(conv2_wen),.out_addr(conv2_waddr),.out_data(conv2_wdata));

    maxpool #(.IN_CH(8),.IN_H(11),.IN_W(11)) pool2_inst (
        .clk(clk),.rst(rst),.start(pool2_start),.done(pool2_done),
        .in_addr(pool2_in_addr),.in_data(pool2_in_data),
        .out_wen(pool2_wen),.out_addr(pool2_waddr),.out_data(pool2_wdata));

    fc_layer fc_inst (
        .clk(clk),.rst(rst),.start(fc_start),.done(fc_done),
        .in_addr(fc_in_addr),.in_data(fc_in_data),
        .out_wen(fc_wen),.out_addr(fc_waddr),.out_data(fc_wdata));

    argmax argmax_inst (
        .clk(clk),.rst(rst),.start(argmax_start),.done(argmax_done),
        .pred(pred),.score_addr(argmax_addr),.score_data(argmax_data));

    // ---------------------------------------------------------------
    // Main sequence
    // ---------------------------------------------------------------
    integer i, errors;

    initial begin
        errors = 0;
        conv1_start<=0; pool1_start<=0; conv2_start<=0;
        pool2_start<=0; fc_start<=0;    argmax_start<=0;

        rst <= 1;
        repeat(4) @(posedge clk); rst <= 0;
        repeat(2) @(posedge clk);

        $display("=== CNN Inference: MNIST test image #0 ===");

        // Conv1
        @(posedge clk); conv1_start <= 1;
        @(posedge clk); conv1_start <= 0;
        @(posedge conv1_done);
        repeat(2) @(posedge clk);
        $display("  conv1 done");

        // Pool1
        @(posedge clk); pool1_start <= 1;
        @(posedge clk); pool1_start <= 0;
        @(posedge pool1_done);
        repeat(2) @(posedge clk);
        $display("  pool1 done");

        // Conv2
        @(posedge clk); conv2_start <= 1;
        @(posedge clk); conv2_start <= 0;
        @(posedge conv2_done);
        repeat(2) @(posedge clk);
        $display("  conv2 done");

        // Pool2
        @(posedge clk); pool2_start <= 1;
        @(posedge clk); pool2_start <= 0;
        @(posedge pool2_done);
        repeat(2) @(posedge clk);
        $display("  pool2 done");

        // FC
        @(posedge clk); fc_start <= 1;
        @(posedge clk); fc_start <= 0;
        @(posedge fc_done);
        repeat(2) @(posedge clk);
        $display("  fc done");

        $display("\nClass scores:");
        for (i = 0; i < 10; i = i + 1)
            $display("  class %0d: %0d", i, score_buf[i]);

        // Argmax
        @(posedge clk); argmax_start <= 1;
        @(posedge clk); argmax_start <= 0;
        @(posedge argmax_done);
        repeat(2) @(posedge clk);

        $display("\nPredicted digit: %0d", pred);
        $display("True label:      7");

        if (pred === 4'd7)
            $display("\n=== INFERENCE CORRECT: predicted 7 ===");
        else begin
            $display("\n=== INFERENCE WRONG: predicted %0d (expected 7) ===", pred);
            $display("(expected with synthetic weights -- use real weights for correct result)");
            errors = errors + 1;
        end

        if (errors == 0) $display("=== ALL TESTS PASSED ===");
        $finish;
    end

endmodule
