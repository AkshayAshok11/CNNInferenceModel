// argmax.sv
//
// Scans 10 FC output scores and outputs the index of the largest value.
// That index is the predicted digit class (0-9).
//
// Reads scores sequentially from a buffer (synchronous read, 1-cycle
// latency). Asserts `done` and outputs `pred` after scanning all 10.
//
// Since FC outputs are already ReLU-clipped to [0,127], argmax is
// straightforward: keep a running max and the index that produced it.
// In the case of ties, the lower class index wins (first-seen).

module argmax (
    input  logic       clk,
    input  logic       rst,
    input  logic       start,
    output logic       done,
    output logic [3:0] pred,    // predicted class 0..9

    // FC score buffer read port (synchronous, 1-cycle latency)
    output logic [3:0] score_addr,
    input  logic signed [7:0] score_data
);

    logic [3:0]        step;          // 0..9
    logic signed [7:0] running_max;
    logic [3:0]        running_idx;

    typedef enum logic [1:0] {
        S_IDLE  = 2'b00,
        S_READ  = 2'b01,   // present address; data arrives next cycle
        S_DONE  = 2'b10
    } state_t;
    state_t state;

    // Prefetch: always address current step so data arrives next cycle
    assign score_addr = step;

    logic valid_d;  // tracks whether score_data is valid this cycle

    always_ff @(posedge clk) begin
        if (rst) begin
            state       <= S_IDLE;
            step        <= 4'd0;
            running_max <= -8'sd128;
            running_idx <= 4'd0;
            valid_d     <= 1'b0;
            done        <= 1'b0;
            pred        <= 4'd0;
        end else begin
            done    <= 1'b0;
            valid_d <= (state == S_READ);

            case (state)
                S_IDLE: begin
                    if (start) begin
                        step        <= 4'd0;
                        running_max <= -8'sd128;
                        running_idx <= 4'd0;
                        state       <= S_READ;
                    end
                end

                S_READ: begin
                    // score_data this cycle is for the address from last cycle
                    if (valid_d) begin
                        if (score_data > running_max) begin
                            running_max <= score_data;
                            running_idx <= step - 4'd1; // addr was step-1 last cycle
                        end
                    end

                    if (step == 4'd9) begin
                        // Presented last address; one more cycle for it to arrive
                        step  <= 4'd10;  // sentinel to trigger done next cycle
                    end else if (step == 4'd10) begin
                        // Final score arrived -- check it and finish
                        if (score_data > running_max) begin
                            running_max <= score_data;
                            running_idx <= 4'd9;
                        end
                        pred  <= (score_data > running_max) ? 4'd9 : running_idx;
                        done  <= 1'b1;
                        state <= S_IDLE;
                    end else begin
                        step <= step + 4'd1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
