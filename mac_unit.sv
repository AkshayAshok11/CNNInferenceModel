// mac_unit.sv
//
// A single multiply-accumulate (MAC) unit.
//
// Each clock cycle, if `valid` is high, it multiplies `in_val` and
// `weight` (both signed 8-bit), and adds the product to a running
// total stored in `acc`. This is the literal hardware equivalent of
// one row of the Python trace from verify_mac.py: multiply, add to
// running sum, repeat.
//
// `clear` resets the accumulator to 0 -- you'll pulse this before
// starting a new 9-cycle MAC sequence for a new output pixel.
//
// Sizing notes:
//   - in_val, weight: signed 8-bit (range -128..127), matching the
//     int8 quantization from quantize.py.
//   - product: signed 16-bit. Worst case magnitude is 127*127=16129
//     (or -128*127=-16256), which fits in 16 bits signed (max ~32767).
//   - acc: signed 20-bit. Summing 9 products of magnitude up to ~16256
//     gives a worst-case magnitude of about 146304, which needs 18
//     bits signed minimum (2^17=131072 is too small, 2^18=262144 is
//     enough). 20 bits gives comfortable headroom for a fully populated
//     larger conv layer (conv2 has 4 input channels * 3x3 = 36 terms
//     per output, not just 9) without needing to resize this module.

module mac_unit (
    input  logic               clk,
    input  logic               rst,      // synchronous reset, clears acc to 0
    input  logic               clear,    // pulse high for 1 cycle to zero acc before a new sequence
    input  logic               valid,    // high when in_val/weight are valid this cycle
    input  logic signed [7:0]  in_val,
    input  logic signed [7:0]  weight,
    output logic signed [19:0] acc
);

    logic signed [15:0] product;

    // Combinational multiply -- the actual multiplier hardware.
    assign product = in_val * weight;

    // Sequential accumulate -- the running total, updated on each
    // clock edge while valid is high.
    always_ff @(posedge clk) begin
        if (rst || clear) begin
            acc <= 20'sd0;
        end else if (valid) begin
            acc <= acc + product;
        end
        // if valid is low and clear/rst are low, acc holds its value
    end

endmodule
