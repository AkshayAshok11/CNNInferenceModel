// fc_layer.sv
//
// Fully connected layer: 200 inputs -> 10 class scores.
//
// For each output class (0..9):
//   score[class] = ReLU(bias[class] + sum_{i=0}^{199} input[i] * weight[class][i])
//
// Reuses mac_unit. Runs 200 MAC steps per class, 10 classes = 2000
// total MACs. Input buffer (pool2 output) has 1-cycle read latency,
// so we use the same prefetch pattern as conv1/conv2: address input[i]
// during step i-1 so it arrives for step i.
//
// Ports:
//   clk, rst, start, done  : control
//   in_addr / in_data       : pool2 output buffer read (synchronous)
//   out_wen / out_addr / out_data : 10 output scores written sequentially

module fc_layer (
    input  logic        clk,
    input  logic        rst,
    input  logic        start,
    output logic        done,

    // Pool2 output read port (synchronous, 1-cycle latency)
    output logic [7:0]  in_addr,   // 0..199
    input  logic signed [7:0] in_data,

    // Output write port: 10 scores, addresses 0..9
    output logic        out_wen,
    output logic [3:0]  out_addr,
    output logic signed [7:0] out_data
);

    // ---------------------------------------------------------------
    // Counters
    // ---------------------------------------------------------------
    logic [3:0]  cls;     // current output class, 0..9
    logic [7:0]  step;    // current input index, 0..199

    // ---------------------------------------------------------------
    // MAC unit signals
    // ---------------------------------------------------------------
    logic        mac_clear, mac_valid;
    logic signed [7:0]  mac_in, mac_weight;
    logic signed [19:0] mac_acc;

    mac_unit mac_inst (
        .clk   (clk),  .rst   (rst),
        .clear (mac_clear),
        .valid (mac_valid),
        .in_val(mac_in),
        .weight(mac_weight),
        .acc   (mac_acc)
    );

    // ---------------------------------------------------------------
    // FC weight ROM
    // ---------------------------------------------------------------
    logic [10:0]       weight_addr;
    logic signed [7:0] weight_out;
    logic [3:0]        bias_addr;
    logic signed [7:0] bias_out;

    fc_rom rom_inst (
        .weight_addr(weight_addr),
        .weight_out (weight_out),
        .bias_addr  (bias_addr),
        .bias_out   (bias_out)
    );

    // ---------------------------------------------------------------
    // Address routing
    // ---------------------------------------------------------------
    // Weight address: class * 200 + step (combinational, 0-cycle latency)
    assign weight_addr = ({7'b0, cls} * 11'd200) + {3'b0, step};
    assign bias_addr   = cls;
    assign mac_weight  = weight_out;
    assign mac_in      = in_data;   // registered, 1-cycle latency

    // Input address: prefetch one step ahead (step+1), same pattern
    // as conv1/conv2. During CLEARING we present step=0's address so
    // it arrives on the first RUNNING cycle.
    logic [7:0] next_step;
    assign next_step = (step == 8'd199) ? 8'd0 : step + 8'd1;
    assign in_addr   = next_step;

    // ---------------------------------------------------------------
    // Bias + ReLU (no scaling needed -- raw int accumulator)
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
    // Registered output
    // ---------------------------------------------------------------
    logic [3:0]        out_addr_r;
    logic signed [7:0] out_data_r;
    assign out_addr = out_addr_r;
    assign out_data = out_data_r;

    // ---------------------------------------------------------------
    // FSM
    // ---------------------------------------------------------------
    typedef enum logic [1:0] {
        S_IDLE    = 2'b00,
        S_CLEAR   = 2'b01,   // one cycle: clear MAC, prefetch input[0]
        S_RUN     = 2'b10,   // 200 cycles: accumulate all inputs
        S_STORE   = 2'b11    // latch + write result, advance class
    } state_t;
    state_t state;

    always_ff @(posedge clk) begin
        if (rst) begin
            state      <= S_IDLE;
            cls        <= 4'd0;
            step       <= 8'd0;
            mac_clear  <= 1'b0;
            mac_valid  <= 1'b0;
            out_wen    <= 1'b0;
            out_addr_r <= 4'd0;
            out_data_r <= 8'sd0;
            done       <= 1'b0;
        end else begin
            mac_clear <= 1'b0;
            mac_valid <= 1'b0;
            out_wen   <= 1'b0;
            done      <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        cls   <= 4'd0;
                        step  <= 8'd0;
                        state <= S_CLEAR;
                    end
                end

                S_CLEAR: begin
                    // Clear accumulator; in_addr already showing next_step=0
                    // so input[0] will arrive next cycle
                    mac_clear <= 1'b1;
                    step      <= 8'd0;
                    state     <= S_RUN;
                end

                S_RUN: begin
                    mac_valid <= 1'b1;

                    if (step == 8'd199) begin
                        // Last step -- one more cycle needed for final
                        // pixel to arrive (same DRAINING idea as kernel_seq)
                        step  <= 8'd0;
                        state <= S_STORE;
                    end else begin
                        step <= step + 8'd1;
                    end
                end

                S_STORE: begin
                    // Final pixel (step 199) arrived this cycle and was
                    // accumulated by mac_unit (mac_valid was high last cycle).
                    // Latch result with current class address, then write.
                    out_addr_r <= cls;
                    out_data_r <= relu_out;
                    out_wen    <= 1'b1;

                    if (cls < 4'd9) begin
                        cls   <= cls + 4'd1;
                        state <= S_CLEAR;
                    end else begin
                        done  <= 1'b1;
                        state <= S_IDLE;
                    end
                end
            endcase
        end
    end

endmodule
