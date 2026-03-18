// =============================================================
// Max Pooling Unit - 2x2 with stride 2
// Target: Zynq-7000 FPGA
// =============================================================

module maxpool_unit #(
    parameter DATA_WIDTH = 8,
    parameter IMG_WIDTH  = 32
)(
    input  logic                         clk,
    input  logic                         rst_n,
    
    input  logic signed [DATA_WIDTH-1:0] data_in,
    input  logic                         data_in_valid,
    
    output logic signed [DATA_WIDTH-1:0] data_out,
    output logic                         data_out_valid
);

    // 2x2 max pooling requires buffering one row
    logic signed [DATA_WIDTH-1:0] line_buffer [0:IMG_WIDTH-1];
    logic [$clog2(IMG_WIDTH)-1:0] col_cnt;
    logic                         row_toggle;  // Even/odd row tracker
    
    // 2x2 window
    logic signed [DATA_WIDTH-1:0] win_00, win_01, win_10, win_11;
    logic signed [DATA_WIDTH-1:0] max_val;
    logic                         pool_valid;

    // Column and row counting
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col_cnt    <= '0;
            row_toggle <= 1'b0;
        end else if (data_in_valid) begin
            if (col_cnt == IMG_WIDTH - 1) begin
                col_cnt    <= '0;
                row_toggle <= ~row_toggle;
            end else begin
                col_cnt <= col_cnt + 1;
            end
        end
    end

    // Line buffer write (first row of each 2x2 block)
    always_ff @(posedge clk) begin
        if (data_in_valid && !row_toggle) begin
            line_buffer[col_cnt] <= data_in;
        end
    end

    // Build 2x2 window on second row + track previous pixel
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            win_00     <= '0;
            win_01     <= '0;
            win_10     <= '0;
            win_11     <= '0;
            pool_valid <= 1'b0;
        end else if (data_in_valid && row_toggle) begin
            if (col_cnt[0]) begin
                // Odd column on odd row = complete 2x2 block
                win_00 <= line_buffer[col_cnt - 1]; // top-left
                win_01 <= line_buffer[col_cnt];      // top-right
                win_10 <= data_in;                    // bottom-right (current)
                // win_11 was set on the previous (even) column
                pool_valid <= 1'b1;
            end else begin
                // Even column on odd row = capture bottom-left
                win_11     <= data_in;
                pool_valid <= 1'b0;
            end
        end else begin
            pool_valid <= 1'b0;
        end
    end

    // Max of 4 values
    function automatic logic signed [DATA_WIDTH-1:0] max4(
        input logic signed [DATA_WIDTH-1:0] a,
        input logic signed [DATA_WIDTH-1:0] b,
        input logic signed [DATA_WIDTH-1:0] c,
        input logic signed [DATA_WIDTH-1:0] d
    );
        logic signed [DATA_WIDTH-1:0] max_ab, max_cd;
        max_ab = (a > b) ? a : b;
        max_cd = (c > d) ? c : d;
        return (max_ab > max_cd) ? max_ab : max_cd;
    endfunction

    // Output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out       <= '0;
            data_out_valid <= 1'b0;
        end else begin
            data_out_valid <= pool_valid;
            if (pool_valid) begin
                data_out <= max4(win_00, win_01, win_10, win_11);
            end
        end
    end

endmodule
