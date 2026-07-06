// conv2.sv
//
// Conv2 layer: 8 filters, 4 input channels, 3x3 kernel, no padding, stride 1.
// Input:  pool1 output, 4 x 13 x 13  signed int8  (676 values)
// Output: 8 x 11 x 11                signed int8  (968 values, ReLU applied)
//
// Key difference from conv1: each output pixel requires 4 input channels
// x 9 kernel positions = 36 MAC operations. The MAC accumulator is cleared
// ONCE at the start of each output pixel (not between channels), so it
// accumulates across all 4 channels continuously.
//
// The kernel_seq module is reused unchanged -- it sequences one 3x3
// kernel (9 steps). This module adds an outer 'ic' loop that runs
// kernel_seq 4 times per output pixel, suppressing mac_clear on
// channels 1..3 so the accumulator keeps running.
//
// Ports mirror conv1 exactly (different sizes):
//   out_addr: 0..967 (10 bits: 2^10=1024 > 968)
//   in_addr:  0..675 (10 bits: 2^10=1024 > 676)

module conv2 (
    input  logic        clk,
    input  logic        rst,
    input  logic        start,
    output logic        done,

    // Pool1 output buffer read port (synchronous: 1-cycle latency)
    output logic [9:0]  in_addr,
    input  logic signed [7:0] in_data,

    // Conv2 output write port
    output logic        out_wen,
    output logic [9:0]  out_addr,
    output logic signed [7:0] out_data
);

    // ---------------------------------------------------------------
    // Submodule signals
    // ---------------------------------------------------------------
    logic       seq_start, seq_done;
    logic [1:0] ky, kx, next_ky, next_kx;
    logic       mac_valid_raw, mac_clear_raw;
    logic       mac_valid;
    logic       mac_clear;

    // mac_valid is used directly (combinational from kernel_seq)
    assign mac_valid = mac_valid_raw;

    logic signed [7:0]  mac_in, mac_weight;
    logic signed [19:0] mac_acc;

    logic [8:0]        weight_addr;
    logic signed [7:0] weight_out;
    logic [2:0]        bias_addr;
    logic signed [7:0] bias_out;

    // ---------------------------------------------------------------
    // Submodule instantiations
    // ---------------------------------------------------------------
    kernel_seq seq_inst (
        .clk      (clk),   .rst      (rst),
        .start    (seq_start),
        .ky       (ky),    .kx       (kx),
        .next_ky  (next_ky), .next_kx(next_kx),
        .mac_valid(mac_valid_raw),
        .mac_clear(mac_clear_raw),
        .done     (seq_done)
    );

    mac_unit mac_inst (
        .clk   (clk),   .rst   (rst),
        .clear (mac_clear),
        .valid (mac_valid),
        .in_val(mac_in),
        .weight(mac_weight),
        .acc   (mac_acc)
    );

    weight_rom2 rom_inst (
        .weight_addr(weight_addr),
        .weight_out (weight_out),
        .bias_addr  (bias_addr),
        .bias_out   (bias_out)
    );

    // ---------------------------------------------------------------
    // Loop counters
    // ---------------------------------------------------------------
    logic [2:0] oc;   // output channel 0..7
    logic [3:0] oy;   // output row     0..10
    logic [3:0] ox;   // output col     0..10
    logic [1:0] ic;   // input channel  0..3

    // ---------------------------------------------------------------
    // FSM
    // ---------------------------------------------------------------
    typedef enum logic [1:0] {
        S_IDLE    = 2'b00,
        S_PIXEL   = 2'b01,   // running kernel_seq (includes DRAINING wait)
        S_NEXT_IC = 2'b10,   // advance ic or latch and store
        S_STORE   = 2'b11    // write output, advance counters
    } state_t;
    state_t state;

    // ---------------------------------------------------------------
    // Address routing
    // ---------------------------------------------------------------

    // Weight address: oc*36 + ic*9 + ky*3 + kx
    assign weight_addr = (9'(oc) * 9'd36)
                       + (9'(ic) * 9'd9)
                       + (9'(ky) * 9'd3)
                       +  9'(kx);
    assign bias_addr   = oc[2:0];

    // Input address: use next_ky/next_kx for one-cycle prefetch.
    // pool1 output is 4 x 13 x 13, flat layout.
    assign in_addr = (10'(ic) * 10'd169)
                   + (10'(oy + {2'b0, next_ky}) * 10'd13)
                   +  10'(ox + {2'b0, next_kx});

    assign mac_in     = in_data;    // registered (1-cycle latency from pool1)
    assign mac_weight = weight_out; // combinational (0-cycle latency)

    // Suppress clear on ic > 0: accumulate across all input channels
    assign mac_clear = mac_clear_raw & (ic == 2'd0);

    // ---------------------------------------------------------------
    // Bias + ReLU
    // ---------------------------------------------------------------
    logic signed [20:0] biased;
    logic signed [20:0] bias_ext;
    logic signed [7:0]  biased_low8;
    logic signed [7:0]  relu_out;

    assign bias_ext    = {{13{bias_out[7]}}, bias_out};
    assign biased      = $signed(mac_acc) + bias_ext;
    assign biased_low8 = biased[7:0];

    always_comb begin
        if (biased <= 0)       relu_out = 8'sd0;
        else if (biased > 127) relu_out = 8'sd127;
        else                   relu_out = biased_low8;
    end

    // ---------------------------------------------------------------
    // Registered output (latch before counters advance)
    // ---------------------------------------------------------------
    logic [9:0]        out_addr_r;
    logic signed [7:0] out_data_r;
    assign out_addr = out_addr_r;
    assign out_data = out_data_r;

    // ---------------------------------------------------------------
    // FSM sequential logic
    // ---------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            state      <= S_IDLE;
            oc         <= 3'd0;
            oy         <= 4'd0;
            ox         <= 4'd0;
            ic         <= 2'd0;
            seq_start  <= 1'b0;
            out_wen    <= 1'b0;
            out_addr_r <= 10'd0;
            out_data_r <= 8'sd0;
            done       <= 1'b0;
        end else begin
            seq_start <= 1'b0;
            out_wen   <= 1'b0;
            done      <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        oc    <= 3'd0;
                        oy    <= 4'd0;
                        ox    <= 4'd0;
                        ic    <= 2'd0;
                        seq_start <= 1'b1;
                        state <= S_PIXEL;
                    end
                end

                S_PIXEL: begin
                    if (seq_done)
                        state <= S_NEXT_IC;
                end

                S_NEXT_IC: begin
                    if (ic < 2'd3) begin
                        // More input channels: start next channel
                        ic        <= ic + 2'd1;
                        seq_start <= 1'b1;
                        state     <= S_PIXEL;
                    end else begin
                        // All 4 channels done. kernel_seq's DRAINING state
                        // already gave mac_acc one extra cycle to include
                        // the final pixel, so relu_out is stable now.
                        out_addr_r <= (10'(oc) * 10'd121)
                                    + (10'(oy) * 10'd11)
                                    +  10'(ox);
                        out_data_r <= relu_out;
                        state      <= S_STORE;
                    end
                end

                S_STORE: begin
                    // out_addr_r and out_data_r were latched in S_NEXT_IC.
                    // Just fire the write enable and advance counters.
                    out_wen <= 1'b1;

                    // Reset input channel for next output pixel
                    ic <= 2'd0;

                    // Advance output position
                    if (ox < 4'd10) begin
                        ox        <= ox + 4'd1;
                        seq_start <= 1'b1;
                        state     <= S_PIXEL;
                    end else if (oy < 4'd10) begin
                        ox        <= 4'd0;
                        oy        <= oy + 4'd1;
                        seq_start <= 1'b1;
                        state     <= S_PIXEL;
                    end else if (oc < 3'd7) begin
                        ox        <= 4'd0;
                        oy        <= 4'd0;
                        oc        <= oc + 3'd1;
                        seq_start <= 1'b1;
                        state     <= S_PIXEL;
                    end else begin
                        state <= S_IDLE;
                        done  <= 1'b1;
                    end
                end
            endcase
        end
    end

endmodule
