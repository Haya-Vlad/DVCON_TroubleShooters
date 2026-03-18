// =============================================================
// CNN Accelerator Top-Level Module
// Integrates: Conv2D → ReLU → MaxPool pipeline
// Target: Zynq-7000 FPGA (USB data transfer, ARM Cortex control)
// =============================================================

module cnn_accelerator_top #(
    parameter DATA_WIDTH  = 8,
    parameter ACC_WIDTH   = 32,
    parameter KERNEL_SIZE = 3,
    parameter INPUT_CH    = 3,
    parameter OUTPUT_CH   = 16,
    parameter IMG_WIDTH   = 32,
    parameter IMG_HEIGHT  = 32
)(
    input  logic        clk,
    input  logic        rst_n,
    
    // AXI-Lite control interface (from ARM Cortex-A9)
    input  logic [31:0] axi_awaddr,
    input  logic        axi_awvalid,
    output logic        axi_awready,
    input  logic [31:0] axi_wdata,
    input  logic [3:0]  axi_wstrb,
    input  logic        axi_wvalid,
    output logic        axi_wready,
    output logic [1:0]  axi_bresp,
    output logic        axi_bvalid,
    input  logic        axi_bready,
    input  logic [31:0] axi_araddr,
    input  logic        axi_arvalid,
    output logic        axi_arready,
    output logic [31:0] axi_rdata,
    output logic [1:0]  axi_rresp,
    output logic        axi_rvalid,
    input  logic        axi_rready,
    
    // Data streaming interface (from USB/UART bridge)
    input  logic signed [DATA_WIDTH-1:0] stream_data_in,
    input  logic                          stream_in_valid,
    output logic                          stream_in_ready,
    
    output logic signed [DATA_WIDTH-1:0] stream_data_out,
    output logic                          stream_out_valid,
    input  logic                          stream_out_ready,
    
    // Status LEDs
    output logic [3:0]  status_leds,
    
    // Interrupt to ARM
    output logic        irq_done
);

    // =========================================================
    // Internal signals
    // =========================================================
    
    // Control signals from AXI-Lite
    logic        ctrl_start;
    logic        ctrl_reset;
    logic [7:0]  ctrl_mode;          // 0=Conv, 1=Conv+ReLU, 2=Conv+ReLU+Pool
    
    // Status signals
    logic        conv_busy, conv_done;
    logic        pipeline_busy;
    
    // Inter-stage connections
    logic signed [ACC_WIDTH-1:0]  conv_out_wide;    // 32-bit accumulator output
    logic signed [DATA_WIDTH-1:0] conv_out, quant_out, relu_out, pool_out;
    logic                          conv_out_valid, quant_out_valid, relu_out_valid, pool_out_valid;
    logic                          conv_out_ready;
    
    // Quantizer scale factor (from AXI register)
    logic [15:0] scale_factor;
    
    // Weight loading signals
    logic signed [DATA_WIDTH-1:0] weight_data;
    logic                          weight_valid, weight_ready;
    
    // Performance counters
    logic [31:0] cycle_count;
    logic [31:0] compute_cycles;
    logic [31:0] data_count;
    
    // =========================================================
    // AXI-Lite Register Interface
    // =========================================================
    
    axi_lite_interface u_axi (
        .clk            (clk),
        .rst_n          (rst_n),
        
        // AXI signals
        .axi_awaddr     (axi_awaddr),
        .axi_awvalid    (axi_awvalid),
        .axi_awready    (axi_awready),
        .axi_wdata      (axi_wdata),
        .axi_wstrb      (axi_wstrb),
        .axi_wvalid     (axi_wvalid),
        .axi_wready     (axi_wready),
        .axi_bresp      (axi_bresp),
        .axi_bvalid     (axi_bvalid),
        .axi_bready     (axi_bready),
        .axi_araddr     (axi_araddr),
        .axi_arvalid    (axi_arvalid),
        .axi_arready    (axi_arready),
        .axi_rdata      (axi_rdata),
        .axi_rresp      (axi_rresp),
        .axi_rvalid     (axi_rvalid),
        .axi_rready     (axi_rready),
        
        // Control outputs
        .ctrl_start     (ctrl_start),
        .ctrl_reset     (ctrl_reset),
        .ctrl_mode      (ctrl_mode),
        .weight_data    (weight_data),
        .weight_valid   (weight_valid),
        .weight_ready   (weight_ready),
        
        // Status inputs
        .status_busy    (pipeline_busy),
        .status_done    (conv_done),
        .cycle_count    (cycle_count),
        .compute_cycles (compute_cycles),
        .data_count     (data_count)
    );

    // =========================================================
    // Conv2D Accelerator
    // =========================================================
    
    conv2d_accelerator #(
        .DATA_WIDTH   (DATA_WIDTH),
        .ACC_WIDTH    (ACC_WIDTH),
        .KERNEL_SIZE  (KERNEL_SIZE),
        .INPUT_CH     (INPUT_CH),
        .OUTPUT_CH    (OUTPUT_CH),
        .IMG_WIDTH    (IMG_WIDTH),
        .IMG_HEIGHT   (IMG_HEIGHT)
    ) u_conv2d (
        .clk          (clk),
        .rst_n        (rst_n & ~ctrl_reset),
        .start        (ctrl_start),
        .done         (conv_done),
        .busy         (conv_busy),
        .input_data   (stream_data_in),
        .input_valid  (stream_in_valid),
        .input_ready  (stream_in_ready),
        .weight_data  (weight_data),
        .weight_valid (weight_valid),
        .weight_ready (weight_ready),
        .output_data  (conv_out_wide),
        .output_valid (conv_out_valid),
        .output_ready (conv_out_ready)
    );

    // =========================================================
    // Quantizer: 32-bit accumulator → INT8
    // =========================================================
    
    quantizer #(
        .DATA_WIDTH   (DATA_WIDTH),
        .FLOAT_WIDTH  (ACC_WIDTH),
        .SCALE_BITS   (16)
    ) u_quantizer (
        .clk            (clk),
        .rst_n          (rst_n),
        .scale_factor   (scale_factor),
        .data_in        (conv_out_wide),
        .data_in_valid  (conv_out_valid),
        .data_out       (quant_out),
        .data_out_valid (quant_out_valid)
    );
    
    // Also provide direct 8-bit truncated path
    assign conv_out = conv_out_wide[DATA_WIDTH-1:0];

    // =========================================================
    // ReLU Unit
    // =========================================================
    
    relu_unit #(
        .DATA_WIDTH (DATA_WIDTH)
    ) u_relu (
        .clk            (clk),
        .rst_n          (rst_n),
        .data_in        (quant_out),
        .data_in_valid  (quant_out_valid),
        .data_out       (relu_out),
        .data_out_valid (relu_out_valid)
    );

    // =========================================================
    // MaxPool Unit
    // =========================================================
    
    maxpool_unit #(
        .DATA_WIDTH (DATA_WIDTH),
        .IMG_WIDTH  (IMG_WIDTH - KERNEL_SIZE + 1) // After convolution
    ) u_maxpool (
        .clk            (clk),
        .rst_n          (rst_n),
        .data_in        (relu_out),
        .data_in_valid  (relu_out_valid),
        .data_out       (pool_out),
        .data_out_valid (pool_out_valid)
    );

    // =========================================================
    // Output Multiplexer (based on mode)
    // =========================================================
    
    always_comb begin
        case (ctrl_mode)
            8'd0: begin // Conv only
                stream_data_out  = conv_out;
                stream_out_valid = conv_out_valid;
                conv_out_ready   = stream_out_ready;
            end
            8'd1: begin // Conv + ReLU
                stream_data_out  = relu_out;
                stream_out_valid = relu_out_valid;
                conv_out_ready   = 1'b1;
            end
            default: begin // Conv + ReLU + MaxPool (full pipeline)
                stream_data_out  = pool_out;
                stream_out_valid = pool_out_valid;
                conv_out_ready   = 1'b1;
            end
        endcase
    end

    // =========================================================
    // Performance Counters
    // =========================================================
    
    assign pipeline_busy = conv_busy;
    assign scale_factor  = 16'h0100;  // Default scale = 1.0 in Q8.8 format
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count    <= '0;
            compute_cycles <= '0;
            data_count     <= '0;
        end else begin
            cycle_count <= cycle_count + 1;
            
            if (conv_busy)
                compute_cycles <= compute_cycles + 1;
                
            if (stream_out_valid && stream_out_ready)
                data_count <= data_count + 1;
                
            if (ctrl_start) begin
                cycle_count    <= '0;
                compute_cycles <= '0;
                data_count     <= '0;
            end
        end
    end

    // =========================================================
    // Status LEDs and Interrupt
    // =========================================================
    
    assign status_leds[0] = pipeline_busy;      // LED0: Processing
    assign status_leds[1] = conv_done;           // LED1: Done
    assign status_leds[2] = stream_in_valid;     // LED2: Data in
    assign status_leds[3] = stream_out_valid;    // LED3: Data out
    
    // Rising edge detection for done interrupt
    logic conv_done_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            conv_done_d <= 1'b0;
        else
            conv_done_d <= conv_done;
    end
    assign irq_done = conv_done & ~conv_done_d;

endmodule
