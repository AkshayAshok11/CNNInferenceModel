// kernel_seq.sv
//
// Kernel sequencer for a 3x3 convolution.
//
// Given a `start` pulse, this module sequences through all 9 kernel
// positions (ky=0..2, kx=0..2), asserting `mac_valid` for 9 cycles
// while outputting the current (ky, kx) so the caller can fetch the
// right input pixel and weight. After the 9th cycle it asserts `done`
// for one cycle, then goes idle.
//
// IMPORTANT timing: `done` fires ONE cycle AFTER the last `mac_valid`.
// This is necessary because input buffers use synchronous (registered)
// reads: the address for step 8 is presented on the last mac_valid
// cycle, but the pixel data arrives one cycle later. The DRAINING state
// gives the MAC unit that extra cycle to accumulate the final pixel
// before done signals the result is ready.
//
// Without this, the caller (conv2) would read mac_acc one cycle too
// early, missing the final accumulation step.
//
// State machine:
//   IDLE     -> CLEARING (on start)
//   CLEARING -> RUNNING  (mac_clear high for one cycle)
//   RUNNING  -> DRAINING (after 9 valid cycles)
//   DRAINING -> IDLE     (done pulses here, one cycle after last valid)

module kernel_seq (
    input  logic       clk,
    input  logic       rst,
    input  logic       start,
    output logic [1:0] ky,        // current kernel row (for weight lookup)
    output logic [1:0] kx,        // current kernel col (for weight lookup)
    output logic [1:0] next_ky,   // next kernel row (for prefetch address)
    output logic [1:0] next_kx,   // next kernel col (for prefetch address)
    output logic       mac_valid,
    output logic       mac_clear,
    output logic       done
);

    typedef enum logic [1:0] {
        IDLE     = 2'b00,
        CLEARING = 2'b01,
        RUNNING  = 2'b10,
        DRAINING = 2'b11    // one extra cycle for final pixel to arrive
    } state_t;

    state_t state;
    logic [3:0] step;

    assign ky = step / 3;
    assign kx = step % 3;

    // next_step: the step AFTER current (wraps to 0 after step 8)
    // used to prefetch the next pixel one cycle early,
    // compensating for the registered (1-cycle latency) input buffer.
    logic [3:0] next_step;
    assign next_step = (step == 4'd8) ? 4'd0 : step + 4'd1;
    assign next_ky = next_step / 3;
    assign next_kx = next_step % 3;

    assign mac_valid = (state == RUNNING);
    assign mac_clear = (state == CLEARING);
    assign done      = (state == DRAINING);  // fires one cycle AFTER last valid

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            step  <= 4'd0;
        end else begin
            case (state)
                IDLE: begin
                    step <= 4'd0;
                    if (start)
                        state <= CLEARING;
                end

                CLEARING: begin
                    state <= RUNNING;
                    step  <= 4'd0;
                end

                RUNNING: begin
                    if (step == 4'd8) begin
                        state <= DRAINING;
                        step  <= 4'd0;
                    end else begin
                        step <= step + 4'd1;
                    end
                end

                DRAINING: begin
                    // done is high this cycle (combinational above).
                    // The final pixel has now been accumulated by mac_unit.
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule