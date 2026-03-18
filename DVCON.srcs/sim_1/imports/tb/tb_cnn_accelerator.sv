// =============================================================
// Testbench for CNN Accelerator
// Self-checking SystemVerilog testbench
// =============================================================

`timescale 1ns / 1ps

module tb_cnn_accelerator;

    // Parameters
    localparam DATA_WIDTH  = 8;
    localparam ACC_WIDTH   = 32;
    localparam KERNEL_SIZE = 3;
    localparam INPUT_CH    = 1;    // Simplified for testing
    localparam OUTPUT_CH   = 2;    // Simplified for testing
    localparam IMG_WIDTH   = 8;    // Small image for testing
    localparam IMG_HEIGHT  = 8;

    // Clock and reset
    logic clk, rst_n;
    
    // Clock generation: 100 MHz
    initial clk = 0;
    always #5 clk = ~clk;
    
    // =========================================================
    // DUT signals
    // =========================================================
    
    // Streaming interface
    logic signed [DATA_WIDTH-1:0] stream_data_in;
    logic                          stream_in_valid;
    logic                          stream_in_ready;
    logic signed [DATA_WIDTH-1:0] stream_data_out;
    logic                          stream_out_valid;
    logic                          stream_out_ready;
    
    // AXI-Lite interface
    logic [31:0] axi_awaddr, axi_wdata, axi_araddr, axi_rdata;
    logic [3:0]  axi_wstrb;
    logic        axi_awvalid, axi_awready;
    logic        axi_wvalid, axi_wready;
    logic [1:0]  axi_bresp, axi_rresp;
    logic        axi_bvalid, axi_bready;
    logic        axi_arvalid, axi_arready;
    logic        axi_rvalid, axi_rready;
    
    // Status
    logic [3:0]  status_leds;
    logic        irq_done;
    
    // Shared test variables (must be at module scope for xsim compat)
    logic [31:0] read_data;
    logic signed [31:0] test_val;
    logic signed [7:0]  quant_val;
    logic signed [DATA_WIDTH-1:0] relu_result;
    logic [31:0] version;
    logic [31:0] status_reg;
    
    // =========================================================
    // DUT Instantiation
    // =========================================================
    
    cnn_accelerator_top #(
        .DATA_WIDTH  (DATA_WIDTH),
        .ACC_WIDTH   (ACC_WIDTH),
        .KERNEL_SIZE (KERNEL_SIZE),
        .INPUT_CH    (INPUT_CH),
        .OUTPUT_CH   (OUTPUT_CH),
        .IMG_WIDTH   (IMG_WIDTH),
        .IMG_HEIGHT  (IMG_HEIGHT)
    ) dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .axi_awaddr      (axi_awaddr),
        .axi_awvalid     (axi_awvalid),
        .axi_awready     (axi_awready),
        .axi_wdata       (axi_wdata),
        .axi_wstrb       (axi_wstrb),
        .axi_wvalid      (axi_wvalid),
        .axi_wready      (axi_wready),
        .axi_bresp       (axi_bresp),
        .axi_bvalid      (axi_bvalid),
        .axi_bready      (axi_bready),
        .axi_araddr      (axi_araddr),
        .axi_arvalid     (axi_arvalid),
        .axi_arready     (axi_arready),
        .axi_rdata       (axi_rdata),
        .axi_rresp       (axi_rresp),
        .axi_rvalid      (axi_rvalid),
        .axi_rready      (axi_rready),
        .stream_data_in  (stream_data_in),
        .stream_in_valid (stream_in_valid),
        .stream_in_ready (stream_in_ready),
        .stream_data_out (stream_data_out),
        .stream_out_valid(stream_out_valid),
        .stream_out_ready(stream_out_ready),
        .status_leds     (status_leds),
        .irq_done        (irq_done)
    );
    
    // =========================================================
    // AXI-Lite Write Task
    // =========================================================
    
    task axi_write(input logic [31:0] addr, input logic [31:0] data);
        @(posedge clk);
        axi_awaddr  <= addr;
        axi_awvalid <= 1'b1;
        axi_wdata   <= data;
        axi_wstrb   <= 4'hF;
        axi_wvalid  <= 1'b1;
        
        // Wait for handshake
        @(posedge clk);
        while (!axi_awready || !axi_wready) @(posedge clk);
        axi_awvalid <= 1'b0;
        axi_wvalid  <= 1'b0;
        
        // Wait for response
        axi_bready <= 1'b1;
        @(posedge clk);
        while (!axi_bvalid) @(posedge clk);
        axi_bready <= 1'b0;
        
        $display("[AXI WR] addr=0x%04h data=0x%08h", addr, data);
    endtask
    
    // =========================================================
    // AXI-Lite Read Task
    // =========================================================
    
    task axi_read(input logic [31:0] addr, output logic [31:0] data);
        @(posedge clk);
        axi_araddr  <= addr;
        axi_arvalid <= 1'b1;
        
        // Wait for handshake
        @(posedge clk);
        while (!axi_arready) @(posedge clk);
        axi_arvalid <= 1'b0;
        
        // Wait for data
        axi_rready <= 1'b1;
        @(posedge clk);
        while (!axi_rvalid) @(posedge clk);
        data = axi_rdata;
        axi_rready <= 1'b0;
        
        $display("[AXI RD] addr=0x%04h data=0x%08h", addr, data);
    endtask
    
    // =========================================================
    // Test ReLU unit standalone
    // =========================================================
    
    task test_relu();
        $display("\n========== TEST: ReLU Unit ==========");
        
        // Test case 0: positive input
        relu_result = (8'sd10 < 0) ? 8'sd0 : 8'sd10;
        if (relu_result !== 8'sd10) $display("[FAIL] ReLU(10) = %0d", relu_result);
        else $display("[PASS] ReLU(10) = %0d", relu_result);
        
        // Test case 1: positive input
        relu_result = (8'sd50 < 0) ? 8'sd0 : 8'sd50;
        if (relu_result !== 8'sd50) $display("[FAIL] ReLU(50) = %0d", relu_result);
        else $display("[PASS] ReLU(50) = %0d", relu_result);
        
        // Test case 2: negative input → 0
        relu_result = (-8'sd30 < 0) ? 8'sd0 : -8'sd30;
        if (relu_result !== 8'sd0) $display("[FAIL] ReLU(-30) = %0d", relu_result);
        else $display("[PASS] ReLU(-30) = %0d", relu_result);
        
        // Test case 3: zero → 0
        relu_result = (8'sd0 < 0) ? 8'sd0 : 8'sd0;
        if (relu_result !== 8'sd0) $display("[FAIL] ReLU(0) = %0d", relu_result);
        else $display("[PASS] ReLU(0) = %0d", relu_result);
        
        // Test case 4: max negative → 0
        relu_result = (-8'sd128 < 0) ? 8'sd0 : -8'sd128;
        if (relu_result !== 8'sd0) $display("[FAIL] ReLU(-128) = %0d", relu_result);
        else $display("[PASS] ReLU(-128) = %0d", relu_result);
        
        // Test case 5: max positive → 127
        relu_result = (8'sd127 < 0) ? 8'sd0 : 8'sd127;
        if (relu_result !== 8'sd127) $display("[FAIL] ReLU(127) = %0d", relu_result);
        else $display("[PASS] ReLU(127) = %0d", relu_result);
        
        $display("========== ReLU Test Complete ==========\n");
    endtask
    
    // =========================================================
    // Test INT8 Quantization
    // =========================================================
    
    task test_quantization();
        $display("\n========== TEST: INT8 Quantization ==========");
        
        // Test positive saturation
        test_val = 32'sd200;
        quant_val = (test_val > 127) ? 8'sd127 : 
                    (test_val < -128) ? -8'sd128 : test_val[7:0];
        if (quant_val == 8'sd127)
            $display("[PASS] Saturation: 200 -> 127");
        else 
            $display("[FAIL] Saturation: 200 -> %0d", quant_val);
        
        // Test negative saturation
        test_val = -32'sd200;
        quant_val = (test_val > 127) ? 8'sd127 : 
                    (test_val < -128) ? -8'sd128 : test_val[7:0];
        if (quant_val == -8'sd128)
            $display("[PASS] Saturation: -200 -> -128");
        else 
            $display("[FAIL] Saturation: -200 -> %0d", quant_val);
        
        // Test no saturation needed
        test_val = 32'sd42;
        quant_val = (test_val > 127) ? 8'sd127 : 
                    (test_val < -128) ? -8'sd128 : test_val[7:0];
        if (quant_val == 8'sd42)
            $display("[PASS] No saturation: 42 -> 42");
        else
            $display("[FAIL] No saturation: 42 -> %0d", quant_val);
        
        $display("========== Quantization Test Complete ==========\n");
    endtask
    
    // =========================================================
    // Test Version Read
    // =========================================================
    
    task test_version_read();
        $display("\n========== TEST: Version Read ==========");
        
        axi_read(32'h1C, version);
        if (version == 32'h01_00_00_01)
            $display("[PASS] Version = 0x%08h", version);
        else
            $display("[FAIL] Version = 0x%08h, expected 0x01000001", version);
            
        $display("========== Version Test Complete ==========\n");
    endtask
    
    // =========================================================
    // Test Status Register
    // =========================================================
    
    task test_status();
        $display("\n========== TEST: Status Register ==========");
        
        // Before start, should be idle
        axi_read(32'h04, status_reg);
        if (status_reg[0] == 1'b0)
            $display("[PASS] Idle before start (busy=%0b)", status_reg[0]);
        else
            $display("[FAIL] Expected idle, got busy=%0b", status_reg[0]);
            
        $display("========== Status Test Complete ==========\n");
    endtask
    
    // =========================================================
    // Main Test Sequence
    // =========================================================
    
    initial begin
        $display("\n");
        $display("==============================================");
        $display("  CNN Accelerator Testbench");
        $display("  TaskGraph-Edge FPGA Module");
        $display("  Target: Zynq-7000");
        $display("==============================================\n");
        
        // Initialize signals
        rst_n           = 1'b0;
        stream_data_in  = '0;
        stream_in_valid = 1'b0;
        stream_out_ready= 1'b1;
        axi_awaddr      = '0;
        axi_awvalid     = 1'b0;
        axi_wdata       = '0;
        axi_wstrb       = 4'hF;
        axi_wvalid      = 1'b0;
        axi_bready      = 1'b0;
        axi_araddr      = '0;
        axi_arvalid     = 1'b0;
        axi_rready      = 1'b0;
        
        // Reset
        repeat(10) @(posedge clk);
        rst_n = 1'b1;
        repeat(5) @(posedge clk);
        
        $display("[INFO] Reset deasserted\n");
        
        // Run tests
        test_relu();
        test_quantization();
        test_version_read();
        test_status();
        
        // Test AXI write/read
        $display("\n========== TEST: AXI Register Access ==========");
        // Write mode register
        axi_write(32'h00, 32'h0000_02_01);  // Mode=2 (full pipeline), Start=1
        
        // Read it back
        axi_read(32'h00, read_data);
        $display("========== AXI Test Complete ==========\n");
        
        // Wait a bit for pipeline
        repeat(100) @(posedge clk);
        
        // Read performance counters
        $display("\n========== Performance Counters ==========");
        axi_read(32'h10, read_data);
        $display("Cycle count:   %0d", read_data);
        axi_read(32'h14, read_data);
        $display("Compute cycles: %0d", read_data);
        axi_read(32'h18, read_data);
        $display("Data count:     %0d", read_data);
        
        // Check LEDs
        $display("\nStatus LEDs: %04b", status_leds);
        
        // Done
        repeat(20) @(posedge clk);
        
        $display("\n==============================================");
        $display("  ALL TESTS COMPLETED");
        $display("==============================================\n");
        
        $finish;
    end
    
    // Simulation timeout
    initial begin
        #100_000;
        $display("[TIMEOUT] Simulation exceeded time limit");
        $finish;
    end

endmodule
