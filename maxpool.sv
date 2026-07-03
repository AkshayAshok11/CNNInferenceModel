// maxpool.sv
//
// 2x2 max pooling, stride 2, applied independently to each channel.
//
// Generic over input dimensions via parameters -- the same module
// is used for both pool1 (4x26x26 -> 4x13x13) and pool2 (8x11x11 -> 8x5x5).
// Note: pool2 input is 11x11 which is odd -- the last row/col is dropped
// (standard "valid" pooling behavior, same as PyTorch's default).
//
// Interface:
//   Input buffer:  caller presents in_addr each cycle; in_data arrives
//                  one cycle later (synchronous read, like input_buf).
//   Output buffer: module writes out_wen/out_addr/out_data each time
//                  a pool window is complete.
//
// Operation:
//   For each channel, for each 2x2 window (non-overlapping, stride 2):
//     - Read 4 pixels: (r,c), (r,c+1), (r+1,c), (r+1,c+1)
//     - Output max of the 4
//
// Timing: 4 cycles to read the 4 pixels (one per cycle, due to
// synchronous memory), 1 cycle to compute max and write output.
// Total cycles = num_channels * out_h * out_w * 5
//
// Parameters:
//   IN_CH   : number of channels (4 for pool1, 8 for pool2)
//   IN_H    : input height  (26 for pool1, 11 for pool2)
//   IN_W    : input width   (26 for pool1, 11 for pool2)
//   IN_SIZE : total input pixels = IN_CH * IN_H * IN_W
//   OUT_H   : output height = IN_H / 2 (integer division)
//   OUT_W   : output width  = IN_W / 2

module maxpool #(
    parameter int IN_CH   = 4,
    parameter int IN_H    = 26,
    parameter int IN_W    = 26,
    parameter int IN_SIZE = IN_CH * IN_H * IN_W,
    parameter int OUT_H   = IN_H / 2,
    parameter int OUT_W   = IN_W / 2,
    parameter int OUT_SIZE = IN_CH * OUT_H * OUT_W,
    parameter int IN_ABITS  = $clog2(IN_SIZE),
    parameter int OUT_ABITS = $clog2(OUT_SIZE)
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    start,
    output logic                    done,

    // Input buffer read port (synchronous: address this cycle, data next)
    output logic [IN_ABITS-1:0]     in_addr,
    input  logic signed [7:0]       in_data,

    // Output buffer write port
    output logic                    out_wen,
    output logic [OUT_ABITS-1:0]    out_addr,
    output logic signed [7:0]       out_data
);

    // ---------------------------------------------------------------
    // Loop counters
    // ---------------------------------------------------------------
    logic [$clog2(IN_CH)-1:0] ch;    // current channel
    logic [$clog2(IN_H)-1:0]  oy;    // output row (0..OUT_H-1)
    logic [$clog2(IN_W)-1:0]  ox;    // output col (0..OUT_W-1)

    // Which of the 4 window pixels we're currently reading (0..3)
    // 0 = (oy*2,   ox*2  )
    // 1 = (oy*2,   ox*2+1)
    // 2 = (oy*2+1, ox*2  )
    // 3 = (oy*2+1, ox*2+1)
    logic [1:0] step;

    // ---------------------------------------------------------------
    // FSM
    // ---------------------------------------------------------------
    typedef enum logic [1:0] {
        S_IDLE    = 2'b00,
        S_READ    = 2'b01,   // reading 4 pixels, one per cycle
        S_WRITE   = 2'b10,   // write max of 4 to output
        S_DONE    = 2'b11
    } state_t;
    state_t state;

    // ---------------------------------------------------------------
    // Max accumulator
    // ---------------------------------------------------------------
    logic signed [7:0] cur_max;

    // ---------------------------------------------------------------
    // Input address calculation
    // ---------------------------------------------------------------
    logic [$clog2(IN_H)-1:0] in_row;
    logic [$clog2(IN_W)-1:0] in_col;
    always_comb begin
        case (step)
            2'd0: begin in_row = oy*2;   in_col = ox*2;   end
            2'd1: begin in_row = oy*2;   in_col = ox*2+1; end
            2'd2: begin in_row = oy*2+1; in_col = ox*2;   end
            2'd3: begin in_row = oy*2+1; in_col = ox*2+1; end
            default: begin in_row = 0; in_col = 0; end
        endcase
    end

    assign in_addr = (IN_ABITS'(ch) * IN_ABITS'(IN_H * IN_W))
                   + (IN_ABITS'(in_row) * IN_ABITS'(IN_W))
                   + IN_ABITS'(in_col);

    // ---------------------------------------------------------------
    // Registered output address and data, latched just before writing
    // so they remain stable while counters advance (same fix as conv1).
    logic [OUT_ABITS-1:0]  out_addr_r;
    logic signed [7:0]     out_data_r;
    assign out_addr = out_addr_r;
    assign out_data = out_data_r;

    // ---------------------------------------------------------------
    // FSM sequential logic
    // ---------------------------------------------------------------

    // One-cycle pipeline: in_data arrives the cycle AFTER in_addr
    // is presented. valid_d tracks whether in_data is meaningful.
    logic       valid_d;

    always_ff @(posedge clk) begin
        if (rst) begin
            state     <= S_IDLE;
            ch        <= '0;
            oy        <= '0;
            ox        <= '0;
            step      <= 2'd0;
            valid_d   <= 1'b0;
            cur_max   <= -8'sd128;
            out_wen   <= 1'b0;
            out_addr_r<= '0;
            out_data_r<= '0;
            done      <= 1'b0;
        end else begin
            out_wen <= 1'b0;
            done    <= 1'b0;
            valid_d <= (state == S_READ);

            case (state)
                S_IDLE: begin
                    if (start) begin
                        ch      <= '0;
                        oy      <= '0;
                        ox      <= '0;
                        step    <= 2'd0;
                        cur_max <= -8'sd128;
                        state   <= S_READ;
                    end
                end

                S_READ: begin
                    // in_data this cycle is the pixel for LAST cycle's
                    // address (one-cycle latency). valid_d tells us
                    // whether last cycle was also S_READ (data is valid).
                    if (valid_d) begin
                        if (in_data > cur_max)
                            cur_max <= in_data;
                    end

                    if (step == 2'd3) begin
                        // Presented the 4th (last) address this cycle.
                        // in_data for that address will arrive next cycle
                        // in S_WRITE, where we do the final max compare.
                        step  <= 2'd0;
                        state <= S_WRITE;
                    end else begin
                        step <= step + 2'd1;
                    end
                end

                S_WRITE: begin
                    // in_data now holds the 4th pixel (step=3).
                    // Do final max compare, latch result, write output.
                    begin
                        logic signed [7:0] final_max;
                        final_max = (in_data > cur_max) ? in_data : cur_max;

                        // Latch addr/data with CURRENT counters before advancing
                        out_addr_r <= (OUT_ABITS'(ch) * OUT_ABITS'(OUT_H * OUT_W))
                                    + (OUT_ABITS'(oy) * OUT_ABITS'(OUT_W))
                                    + OUT_ABITS'(ox);
                        out_data_r <= final_max;
                        out_wen    <= 1'b1;

                        // Reset max for next window
                        cur_max <= -8'sd128;
                    end

                    // Advance counters
                    if (ox < OUT_W - 1) begin
                        ox    <= ox + 1;
                        state <= S_READ;
                    end else if (oy < OUT_H - 1) begin
                        ox    <= '0;
                        oy    <= oy + 1;
                        state <= S_READ;
                    end else if (ch < IN_CH - 1) begin
                        ox    <= '0;
                        oy    <= '0;
                        ch    <= ch + 1;
                        state <= S_READ;
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
