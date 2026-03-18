// =============================================================
// INT8 Quantizer - FP32 to INT8 conversion
// Target: Zynq-7000 FPGA
// =============================================================

module quantizer #(
    parameter DATA_WIDTH   = 8,
    parameter FLOAT_WIDTH  = 32,
    parameter SCALE_BITS   = 16      // Fixed-point scale factor bits
)(
    input  logic                          clk,
    input  logic                          rst_n,
    
    // Configuration
    input  logic [SCALE_BITS-1:0]         scale_factor,  // Q8.8 fixed-point scale
    
    // Input (wider precision)
    input  logic signed [FLOAT_WIDTH-1:0] data_in,
    input  logic                          data_in_valid,
    
    // Output (INT8)
    output logic signed [DATA_WIDTH-1:0]  data_out,
    output logic                          data_out_valid
);

    // Intermediate computation
    logic signed [FLOAT_WIDTH+SCALE_BITS-1:0] scaled_value;
    logic signed [FLOAT_WIDTH-1:0]            rounded_value;
    
    // Pipeline stage 1: Multiply by scale factor
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scaled_value   <= '0;
            data_out_valid <= 1'b0;
        end else begin
            data_out_valid <= data_in_valid;
            if (data_in_valid) begin
                scaled_value <= data_in * $signed({1'b0, scale_factor});
            end
        end
    end
    
    // Extract and round (shift right by SCALE_BITS/2 for Q8.8 format)
    assign rounded_value = scaled_value[FLOAT_WIDTH+SCALE_BITS/2-1:SCALE_BITS/2];
    
    // Saturating clamp to INT8 range [-128, 127]
    always_comb begin
        if (rounded_value > 127)
            data_out = 8'sd127;
        else if (rounded_value < -128)
            data_out = -8'sd128;
        else
            data_out = rounded_value[DATA_WIDTH-1:0];
    end

endmodule
