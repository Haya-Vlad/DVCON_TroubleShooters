// =============================================================
// Verilog Wrapper for CNN Accelerator
// Required because Vivado Block Design does not support
// SystemVerilog (.sv) as a module reference.
// =============================================================

module cnn_accelerator_wrapper #(
    parameter DATA_WIDTH  = 8,
    parameter ACC_WIDTH   = 32,
    parameter KERNEL_SIZE = 3,
    parameter INPUT_CH    = 3,
    parameter OUTPUT_CH   = 16,
    parameter IMG_WIDTH   = 32,
    parameter IMG_HEIGHT  = 32
)(
    input  wire        clk,
    input  wire        rst_n,

    // AXI-Lite control interface (from ARM Cortex-A9)
    input  wire [31:0] axi_awaddr,
    input  wire        axi_awvalid,
    output wire        axi_awready,
    input  wire [31:0] axi_wdata,
    input  wire [3:0]  axi_wstrb,
    input  wire        axi_wvalid,
    output wire        axi_wready,
    output wire [1:0]  axi_bresp,
    output wire        axi_bvalid,
    input  wire        axi_bready,
    input  wire [31:0] axi_araddr,
    input  wire        axi_arvalid,
    output wire        axi_arready,
    output wire [31:0] axi_rdata,
    output wire [1:0]  axi_rresp,
    output wire        axi_rvalid,
    input  wire        axi_rready,

    // Data streaming interface
    input  wire [DATA_WIDTH-1:0] stream_data_in,
    input  wire                  stream_in_valid,
    output wire                  stream_in_ready,

    output wire [DATA_WIDTH-1:0] stream_data_out,
    output wire                  stream_out_valid,
    input  wire                  stream_out_ready,

    // Status LEDs
    output wire [3:0]  status_leds,

    // Interrupt to ARM
    output wire        irq_done
);

    // Instantiate the SystemVerilog module
    cnn_accelerator_top #(
        .DATA_WIDTH  (DATA_WIDTH),
        .ACC_WIDTH   (ACC_WIDTH),
        .KERNEL_SIZE (KERNEL_SIZE),
        .INPUT_CH    (INPUT_CH),
        .OUTPUT_CH   (OUTPUT_CH),
        .IMG_WIDTH   (IMG_WIDTH),
        .IMG_HEIGHT  (IMG_HEIGHT)
    ) u_core (
        .clk              (clk),
        .rst_n            (rst_n),
        .axi_awaddr       (axi_awaddr),
        .axi_awvalid      (axi_awvalid),
        .axi_awready      (axi_awready),
        .axi_wdata        (axi_wdata),
        .axi_wstrb        (axi_wstrb),
        .axi_wvalid       (axi_wvalid),
        .axi_wready       (axi_wready),
        .axi_bresp        (axi_bresp),
        .axi_bvalid       (axi_bvalid),
        .axi_bready       (axi_bready),
        .axi_araddr       (axi_araddr),
        .axi_arvalid      (axi_arvalid),
        .axi_arready      (axi_arready),
        .axi_rdata        (axi_rdata),
        .axi_rresp        (axi_rresp),
        .axi_rvalid       (axi_rvalid),
        .axi_rready       (axi_rready),
        .stream_data_in   (stream_data_in),
        .stream_in_valid  (stream_in_valid),
        .stream_in_ready  (stream_in_ready),
        .stream_data_out  (stream_data_out),
        .stream_out_valid (stream_out_valid),
        .stream_out_ready (stream_out_ready),
        .status_leds      (status_leds),
        .irq_done         (irq_done)
    );

endmodule
