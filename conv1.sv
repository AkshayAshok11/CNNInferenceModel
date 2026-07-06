// conv1.sv
//
// Full conv1 layer: 4 filters, 3x3 kernel, no padding, stride 1.
// Input:  28x28x1  signed int8  (from input_buf)
// Output: 26x26x4  signed int8  (ReLU applied, stored in output_buf)
//
// Instantiates and connects:
//   - kernel_seq  : sequences through the 9 kernel positions per pixel
//   - mac_unit    : accumulates one dot product
//   - weight_rom  : serves weight[oc][ky][kx] and bias[oc]
//   - input_buf   : serves pixel[row][col]
//
// Outer loop (driven by this module's FSM):
//   for oc = 0..3       (4 output channels)
//     for oy = 0..25    (26 output rows)
//       for ox = 0..25  (26 output cols)
//         fire start -> kernel_seq -> 9 MACs -> bias+ReLU -> store
//
// Timing note: input_buf has 1-cycle read latency (synchronous).
// weight_rom is combinational (0-cycle latency).
// During CLEARING state, we pre-load the address for pixel[oy][ox]
// so its value is ready on step 0 of RUNNING.
// During each RUNNING step i, we address pixel[oy + next_ky][ox + next_kx]
// so it arrives in time for step i+1.
//
// Ports:
//   clk, rst   : clock and synchronous reset
//   start      : pulse to begin processing one full image
//   done       : pulses high when all 2704 output pixels are written
//   -- output memory write port (caller reads the 26x26x4 result) --
//   out_wen    : write enable
//   out_addr   : write address (oc*676 + oy*26 + ox), 0..2703
//   out_data   : signed 8-bit output pixel after bias+ReLU

module conv1 (
    input  logic        clk,
    input  logic        rst,
    input  logic        start,
    output logic        done,

    // Output write port -- connect to an output buffer in the testbench
    output logic        out_wen,
    output logic [11:0] out_addr,   // 0..2703 (12 bits: 2^12=4096 > 2704)
    output logic signed [7:0] out_data
);

    // ---------------------------------------------------------------
    // Submodule signals
    // ---------------------------------------------------------------

    // kernel_seq
    logic       seq_start, seq_done;
    logic [1:0] ky, kx, next_ky, next_kx;
    logic       mac_valid, mac_clear;

    // mac_unit
    logic signed [7:0]  mac_in, mac_weight;
    logic signed [19:0] mac_acc;

    // weight_rom
    logic [5:0]        weight_addr;
    logic signed [7:0] weight_out;
    logic [1:0]        bias_addr;
    logic signed [7:0] bias_out;

    // input_buf
    logic [4:0]        pix_row, pix_col;
    logic signed [7:0] pixel_out;

    // ---------------------------------------------------------------
    // Submodule instantiations
    // ---------------------------------------------------------------

    kernel_seq seq_inst (
        .clk      (clk),
        .rst      (rst),
        .start    (seq_start),
        .ky       (ky),
        .kx       (kx),
        .next_ky  (next_ky),
        .next_kx  (next_kx),
        .mac_valid(mac_valid),
        .mac_clear(mac_clear),
        .done     (seq_done)
    );

    mac_unit mac_inst (
        .clk   (clk),
        .rst   (rst),
        .clear (mac_clear),
        .valid (mac_valid),
        .in_val(mac_in),
        .weight(mac_weight),
        .acc   (mac_acc)
    );

    weight_rom rom_inst (
        .weight_addr(weight_addr),
        .weight_out (weight_out),
        .bias_addr  (bias_addr),
        .bias_out   (bias_out)
    );

    input_buf buf_inst (
        .clk      (clk),
        .row      (pix_row),
        .col      (pix_col),
        .pixel_out(pixel_out)
    );

    // ---------------------------------------------------------------
    // Outer loop counters
    // ---------------------------------------------------------------

    logic [1:0] oc;        // output channel, 0..3
    logic [4:0] oy;        // output row,     0..25
    logic [4:0] ox;        // output col,     0..25

    // ---------------------------------------------------------------
    // Outer FSM
    // ---------------------------------------------------------------

    typedef enum logic [1:0] {
        S_IDLE    = 2'b00,
        S_PIXEL   = 2'b01,   // waiting for kernel_seq to finish one pixel
        S_STORE   = 2'b10,   // bias + ReLU + write output, advance counters
        S_DONE    = 2'b11
    } state_t;

    state_t state;

    // ---------------------------------------------------------------
    // Address and data routing
    // ---------------------------------------------------------------

    // Weight ROM: addressed combinationally from current oc, ky, kx
    // oc*9 + ky*3 + kx (all 6-bit arithmetic)
    assign weight_addr = (6'(oc) * 6'd9) + (6'(ky) * 6'd3) + 6'(kx);
    assign bias_addr   = oc;

    // Feed weight to MAC (combinational, same-cycle)
    assign mac_weight = weight_out;

    // Feed pixel to MAC (registered, one-cycle latency from input_buf)
    assign mac_in = pixel_out;

    // Input buffer address: use NEXT step's ky/kx for prefetch.
    // input_buf has 1-cycle registered read latency: address presented
    // this cycle → pixel available NEXT cycle. By addressing one step
    // ahead, pixel[step_N] arrives exactly when step_N's MAC fires.
    assign pix_row = oy + {3'b0, next_ky};
    assign pix_col = ox + {3'b0, next_kx};

    // Output address and data: registered at the moment seq_done fires
    // (end of S_PIXEL), so they remain stable during S_STORE even as
    // oc/oy/ox advance to the next pixel position.
    logic [11:0]       out_addr_r;
    logic signed [7:0] out_data_r;
    assign out_addr = out_addr_r;
    assign out_data = out_data_r;

    // Bias + ReLU: computed combinationally from mac_acc + bias
    // mac_acc is 20-bit signed; bias_out is 8-bit signed.
    // Sign-extend bias_out to 21 bits by replicating its MSB 13 times.
    logic signed [20:0] biased;
    logic signed [20:0] bias_extended;
    assign bias_extended = {{13{bias_out[7]}}, bias_out};
    assign biased = $signed(mac_acc) + bias_extended;

    // ReLU: if biased <= 0, output 0; else clip to 127 if overflow
    logic signed [7:0] relu_out;
    logic signed [7:0] biased_low8;
    assign biased_low8 = biased[7:0];   // extract low byte outside always_comb
    always_comb begin
        if (biased <= 0)
            relu_out = 8'sd0;
        else if (biased > 127)
            relu_out = 8'sd127;
        else
            relu_out = biased_low8;
    end

    // relu_out feeds into out_data_r (latched when seq_done fires)

    // ---------------------------------------------------------------
    // Outer FSM sequential logic
    // ---------------------------------------------------------------

    always_ff @(posedge clk) begin
        if (rst) begin
            state     <= S_IDLE;
            oc        <= 2'd0;
            oy        <= 5'd0;
            ox        <= 5'd0;
            seq_start <= 1'b0;
            out_wen   <= 1'b0;
            done      <= 1'b0;
        end else begin
            // Default: deassert one-cycle pulses
            seq_start <= 1'b0;
            out_wen   <= 1'b0;
            done      <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        oc    <= 2'd0;
                        oy    <= 5'd0;
                        ox    <= 5'd0;
                        seq_start <= 1'b1;
                        state <= S_PIXEL;
                    end
                end

                S_PIXEL: begin
                    if (seq_done) begin
                        // kernel_seq finished this pixel's 9 MACs.
                        // Latch the current address and relu result now,
                        // before oc/oy/ox advance in S_STORE.
                        out_addr_r <= (12'(oc) * 12'd676) + (12'(oy) * 12'd26) + 12'(ox);
                        out_data_r <= relu_out;
                        state      <= S_STORE;
                    end
                end

                S_STORE: begin
                    // Write the latched bias+ReLU result.
                    // out_addr_r/out_data_r are stable from S_PIXEL latch.
                    out_wen <= 1'b1;

                    // Now advance to next output position
                    if (ox < 5'd25) begin
                        ox        <= ox + 5'd1;
                        seq_start <= 1'b1;
                        state     <= S_PIXEL;
                    end else if (oy < 5'd25) begin
                        ox        <= 5'd0;
                        oy        <= oy + 5'd1;
                        seq_start <= 1'b1;
                        state     <= S_PIXEL;
                    end else if (oc < 2'd3) begin
                        ox        <= 5'd0;
                        oy        <= 5'd0;
                        oc        <= oc + 2'd1;
                        seq_start <= 1'b1;
                        state     <= S_PIXEL;
                    end else begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule