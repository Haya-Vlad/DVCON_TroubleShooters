// =============================================================
// Conv2D Accelerator - Systolic Array Design
// INT8 Multiply-Accumulate for CNN Feature Extraction
// Target: Zynq-7000 FPGA (Artix-7 fabric)
// =============================================================

module conv2d_accelerator #(
    parameter DATA_WIDTH    = 8,          // INT8
    parameter ACC_WIDTH     = 32,         // Accumulator width
    parameter KERNEL_SIZE   = 3,          // 3x3 convolution
    parameter INPUT_CH      = 3,          // Input channels
    parameter OUTPUT_CH     = 16,         // Output channels
    parameter IMG_WIDTH     = 32,         // Feature map width
    parameter IMG_HEIGHT    = 32          // Feature map height
)(
    input  logic                        clk,
    input  logic                        rst_n,
    
    // Control interface
    input  logic                        start,
    output logic                        done,
    output logic                        busy,
    
    // Input data interface
    input  logic signed [DATA_WIDTH-1:0]  input_data,
    input  logic                          input_valid,
    output logic                          input_ready,
    
    // Weight data interface  
    input  logic signed [DATA_WIDTH-1:0]  weight_data,
    input  logic                          weight_valid,
    output logic                          weight_ready,
    
    // Output data interface (full accumulator width for external quantizer)
    output logic signed [ACC_WIDTH-1:0]    output_data,
    output logic                           output_valid,
    input  logic                           output_ready
);

    // =========================================================
    // Internal signals
    // =========================================================
    
    // State machine
    typedef enum logic [2:0] {
        IDLE        = 3'b000,
        LOAD_WEIGHTS = 3'b001,
        COMPUTE     = 3'b010,
        OUTPUT_DATA  = 3'b011,
        DONE_STATE  = 3'b100
    } state_t;
    
    state_t state, next_state;
    
    // Weight memory: stores KERNEL_SIZE x KERNEL_SIZE x INPUT_CH x OUTPUT_CH weights
    localparam WEIGHT_DEPTH = KERNEL_SIZE * KERNEL_SIZE * INPUT_CH * OUTPUT_CH;
    logic signed [DATA_WIDTH-1:0] weight_mem [0:WEIGHT_DEPTH-1];
    logic [$clog2(WEIGHT_DEPTH)-1:0] weight_addr;
    
    // Input line buffer (3 lines for 3x3 convolution)
    logic signed [DATA_WIDTH-1:0] line_buffer [0:2][0:IMG_WIDTH-1];
    logic [$clog2(IMG_WIDTH)-1:0] col_cnt;
    logic [1:0] row_in_buf;
    
    // Convolution window
    logic signed [DATA_WIDTH-1:0] conv_window [0:KERNEL_SIZE-1][0:KERNEL_SIZE-1];
    
    // MAC array
    logic signed [ACC_WIDTH-1:0] accumulators [0:OUTPUT_CH-1];
    logic [$clog2(OUTPUT_CH)-1:0] out_ch_cnt;
    logic [$clog2(INPUT_CH)-1:0]  in_ch_cnt;
    logic [$clog2(KERNEL_SIZE)-1:0] ky_cnt, kx_cnt;
    
    // Position tracking
    logic [$clog2(IMG_HEIGHT)-1:0] row_cnt;
    logic [$clog2(IMG_WIDTH)-1:0]  out_col_cnt;
    logic [$clog2(IMG_HEIGHT)-1:0] out_row_cnt;
    
    // Output scaling (INT8 quantization)
    logic signed [ACC_WIDTH-1:0] scaled_output;
    
    // =========================================================
    // State Machine
    // =========================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end
    
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (start)
                    next_state = LOAD_WEIGHTS;
            end
            LOAD_WEIGHTS: begin
                if (weight_addr == WEIGHT_DEPTH - 1 && weight_valid)
                    next_state = COMPUTE;
            end
            COMPUTE: begin
                if (out_row_cnt == IMG_HEIGHT - KERNEL_SIZE && 
                    out_col_cnt == IMG_WIDTH - KERNEL_SIZE &&
                    out_ch_cnt == OUTPUT_CH - 1)
                    next_state = OUTPUT_DATA;
            end
            OUTPUT_DATA: begin
                if (output_valid && output_ready)
                    next_state = DONE_STATE;
            end
            DONE_STATE: begin
                next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end
    
    // =========================================================
    // Weight Loading
    // =========================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_addr <= '0;
        end else if (state == LOAD_WEIGHTS && weight_valid) begin
            weight_mem[weight_addr] <= weight_data;
            weight_addr <= weight_addr + 1;
        end else if (state == IDLE) begin
            weight_addr <= '0;
        end
    end
    
    assign weight_ready = (state == LOAD_WEIGHTS);
    
    // =========================================================
    // MAC Unit - Core Computation
    // =========================================================
    
    // Single MAC operation
    function automatic logic signed [ACC_WIDTH-1:0] mac_op(
        input logic signed [DATA_WIDTH-1:0] a,
        input logic signed [DATA_WIDTH-1:0] b,
        input logic signed [ACC_WIDTH-1:0]  acc
    );
        return acc + (a * b);
    endfunction
    
    // Main compute logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < OUTPUT_CH; i++)
                accumulators[i] <= '0;
            kx_cnt <= '0;
            ky_cnt <= '0;
            in_ch_cnt <= '0;
            out_ch_cnt <= '0;
        end else if (state == COMPUTE && input_valid) begin
            // Perform MAC for current position
            automatic logic [$clog2(WEIGHT_DEPTH)-1:0] w_idx;
            w_idx = out_ch_cnt * (KERNEL_SIZE * KERNEL_SIZE * INPUT_CH) +
                    in_ch_cnt * (KERNEL_SIZE * KERNEL_SIZE) +
                    ky_cnt * KERNEL_SIZE + kx_cnt;
            
            accumulators[out_ch_cnt] <= mac_op(
                conv_window[ky_cnt][kx_cnt],
                weight_mem[w_idx],
                accumulators[out_ch_cnt]
            );
            
            // Counter management
            if (kx_cnt == KERNEL_SIZE - 1) begin
                kx_cnt <= '0;
                if (ky_cnt == KERNEL_SIZE - 1) begin
                    ky_cnt <= '0;
                    if (in_ch_cnt == INPUT_CH - 1) begin
                        in_ch_cnt <= '0;
                        if (out_ch_cnt == OUTPUT_CH - 1) begin
                            out_ch_cnt <= '0;
                        end else begin
                            out_ch_cnt <= out_ch_cnt + 1;
                        end
                    end else begin
                        in_ch_cnt <= in_ch_cnt + 1;
                    end
                end else begin
                    ky_cnt <= ky_cnt + 1;
                end
            end else begin
                kx_cnt <= kx_cnt + 1;
            end
        end
    end
    
    // =========================================================
    // Output Quantization (INT8)
    // =========================================================
    
    // Saturating clip to INT8 range
    function automatic logic signed [DATA_WIDTH-1:0] saturate_int8(
        input logic signed [ACC_WIDTH-1:0] value
    );
        if (value > 127)
            return 8'sd127;
        else if (value < -128)
            return -8'sd128;
        else
            return value[DATA_WIDTH-1:0];
    endfunction
    
    assign scaled_output = accumulators[out_ch_cnt];
    assign output_data = scaled_output;  // Full 32-bit output for external quantizer
    
    // =========================================================
    // Control signals
    // =========================================================
    
    assign busy = (state != IDLE && state != DONE_STATE);
    assign done = (state == DONE_STATE);
    assign input_ready = (state == COMPUTE);
    assign output_valid = (state == OUTPUT_DATA);

endmodule
