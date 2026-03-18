// =============================================================
// ReLU Activation Unit - Pipelined
// Target: Zynq-7000 FPGA
// =============================================================

module relu_unit #(
    parameter DATA_WIDTH = 8
)(
    input  logic                         clk,
    input  logic                         rst_n,
    
    input  logic signed [DATA_WIDTH-1:0] data_in,
    input  logic                         data_in_valid,
    
    output logic signed [DATA_WIDTH-1:0] data_out,
    output logic                         data_out_valid
);

    // Single-cycle pipelined ReLU: max(0, x)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out       <= '0;
            data_out_valid <= 1'b0;
        end else begin
            data_out_valid <= data_in_valid;
            if (data_in_valid) begin
                data_out <= (data_in[DATA_WIDTH-1]) ? '0 : data_in; // if negative, output 0
            end
        end
    end

endmodule
