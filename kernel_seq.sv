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
// The caller's job (the conv1 layer above this) is to:
//   1. Assert `start` for one cycle to begin a new output pixel.
//   2. Each cycle that `mac_valid` is high, look up input[oy+ky][ox+kx]
//      and weight[oc][ky][kx] and feed them to the MAC unit.
//   3. When `done` goes high, read the MAC accumulator, add bias,
//      apply ReLU, and store the result.
//
// This module also drives `mac_clear` high on the cycle BEFORE the
// first `mac_valid`, so the MAC unit's accumulator is guaranteed to
// be zero before the 9 new products start accumulating. The clear
// happens combinationally one cycle ahead via the CLEARING state.
//
// State machine:
//   IDLE     -> CLEARING (on start)
//   CLEARING -> RUNNING  (mac_clear high for this one cycle)
//   RUNNING  -> IDLE     (after 9 valid cycles; done pulses)

module kernel_seq (
    input  logic       clk,
    input  logic       rst,
    input  logic       start,     // pulse high for 1 cycle to begin
    output logic [1:0] ky,        // current kernel row   (0..2)
    output logic [1:0] kx,        // current kernel col   (0..2)
    output logic       mac_valid, // high while MAC should accumulate
    output logic       mac_clear, // high for 1 cycle to zero MAC acc
    output logic       done       // high for 1 cycle when all 9 done
);

    // State encoding
    typedef enum logic [1:0] {
        IDLE     = 2'b00,
        CLEARING = 2'b01,   // one cycle: clear the MAC accumulator
        RUNNING  = 2'b10    // nine cycles: feed input/weight pairs
    } state_t;

    state_t state;

    // Flat step counter: 0..8 maps to (ky=0,kx=0)..(ky=2,kx=2)
    logic [3:0] step; // needs to count 0..8, so 4 bits

    // Decode step -> (ky, kx)
    assign ky = step / 3;   // 0,0,0,1,1,1,2,2,2
    assign kx = step % 3;   // 0,1,2,0,1,2,0,1,2

    assign mac_valid = (state == RUNNING);
    assign mac_clear = (state == CLEARING);
    assign done      = (state == RUNNING) && (step == 4'd8);

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
                    // Hold for one cycle so the MAC acc resets,
                    // then immediately begin running.
                    state <= RUNNING;
                    step  <= 4'd0;
                end

                RUNNING: begin
                    if (step == 4'd8) begin
                        // Just finished the 9th MAC (step 8).
                        // done is already asserted combinationally above.
                        state <= IDLE;
                        step  <= 4'd0;
                    end else begin
                        step <= step + 4'd1;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
