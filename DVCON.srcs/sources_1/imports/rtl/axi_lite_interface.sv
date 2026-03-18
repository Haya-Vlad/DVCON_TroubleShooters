// =============================================================
// AXI4-Lite Slave Interface for CPU ↔ FPGA Communication
// Register map for controlling CNN accelerator from ARM Cortex-A9
// Target: Zynq-7000 PS-PL interface
// =============================================================
//
// Register Map:
// 0x00: CTRL     (W)  - Bit 0: Start, Bit 1: Reset, Bits [15:8]: Mode
// 0x04: STATUS   (R)  - Bit 0: Busy, Bit 1: Done
// 0x08: WEIGHT   (W)  - Weight data (INT8 in bits [7:0])
// 0x0C: WEIGHT_CTRL (W) - Bit 0: Weight valid
// 0x10: CYCLE_CNT (R) - Total cycle count
// 0x14: COMP_CNT  (R) - Compute cycle count  
// 0x18: DATA_CNT  (R) - Output data count
// 0x1C: VERSION   (R) - IP version (0x01_00_00_01)
// =============================================================

module axi_lite_interface #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH_AXI = 32
)(
    input  logic                          clk,
    input  logic                          rst_n,
    
    // AXI4-Lite Write Address Channel
    input  logic [ADDR_WIDTH-1:0]         axi_awaddr,
    input  logic                          axi_awvalid,
    output logic                          axi_awready,
    
    // AXI4-Lite Write Data Channel
    input  logic [DATA_WIDTH_AXI-1:0]     axi_wdata,
    input  logic [3:0]                    axi_wstrb,
    input  logic                          axi_wvalid,
    output logic                          axi_wready,
    
    // AXI4-Lite Write Response Channel
    output logic [1:0]                    axi_bresp,
    output logic                          axi_bvalid,
    input  logic                          axi_bready,
    
    // AXI4-Lite Read Address Channel
    input  logic [ADDR_WIDTH-1:0]         axi_araddr,
    input  logic                          axi_arvalid,
    output logic                          axi_arready,
    
    // AXI4-Lite Read Data Channel
    output logic [DATA_WIDTH_AXI-1:0]     axi_rdata,
    output logic [1:0]                    axi_rresp,
    output logic                          axi_rvalid,
    input  logic                          axi_rready,
    
    // Control outputs to accelerator
    output logic                          ctrl_start,
    output logic                          ctrl_reset,
    output logic [7:0]                    ctrl_mode,
    output logic signed [7:0]             weight_data,
    output logic                          weight_valid,
    input  logic                          weight_ready,
    
    // Status inputs from accelerator
    input  logic                          status_busy,
    input  logic                          status_done,
    input  logic [31:0]                   cycle_count,
    input  logic [31:0]                   compute_cycles,
    input  logic [31:0]                   data_count
);

    // IP version
    localparam VERSION = 32'h01_00_00_01;
    
    // Internal registers
    logic [DATA_WIDTH_AXI-1:0] ctrl_reg;
    logic [DATA_WIDTH_AXI-1:0] weight_reg;
    logic [DATA_WIDTH_AXI-1:0] weight_ctrl_reg;
    
    // Write FSM
    typedef enum logic [1:0] {
        WR_IDLE  = 2'b00,
        WR_DATA  = 2'b01,
        WR_RESP  = 2'b10
    } wr_state_t;
    
    wr_state_t wr_state;
    logic [ADDR_WIDTH-1:0] wr_addr_reg;
    
    // Read FSM
    typedef enum logic [1:0] {
        RD_IDLE = 2'b00,
        RD_DATA = 2'b01
    } rd_state_t;
    
    rd_state_t rd_state;
    logic [ADDR_WIDTH-1:0] rd_addr_reg;

    // =========================================================
    // Write FSM
    // =========================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_state    <= WR_IDLE;
            axi_awready <= 1'b0;
            axi_wready  <= 1'b0;
            axi_bvalid  <= 1'b0;
            axi_bresp   <= 2'b00;
            wr_addr_reg <= '0;
            ctrl_reg    <= '0;
            weight_reg  <= '0;
            weight_ctrl_reg <= '0;
        end else begin
            case (wr_state)
                WR_IDLE: begin
                    axi_bvalid <= 1'b0;
                    if (axi_awvalid && axi_wvalid) begin
                        axi_awready <= 1'b1;
                        axi_wready  <= 1'b1;
                        wr_addr_reg <= axi_awaddr;
                        wr_state    <= WR_DATA;
                    end
                end
                
                WR_DATA: begin
                    axi_awready <= 1'b0;
                    axi_wready  <= 1'b0;
                    
                    // Decode address and write to register
                    case (wr_addr_reg[7:0])
                        8'h00: ctrl_reg        <= axi_wdata;  // CTRL
                        8'h08: weight_reg      <= axi_wdata;  // WEIGHT
                        8'h0C: weight_ctrl_reg <= axi_wdata;  // WEIGHT_CTRL
                        default: ; // ignore writes to read-only regs
                    endcase
                    
                    axi_bvalid <= 1'b1;
                    axi_bresp  <= 2'b00; // OKAY
                    wr_state   <= WR_RESP;
                end
                
                WR_RESP: begin
                    if (axi_bready) begin
                        axi_bvalid <= 1'b0;
                        wr_state   <= WR_IDLE;
                    end
                end
                
                default: wr_state <= WR_IDLE;
            endcase
        end
    end
    
    // =========================================================
    // Read FSM
    // =========================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_state    <= RD_IDLE;
            axi_arready <= 1'b0;
            axi_rvalid  <= 1'b0;
            axi_rresp   <= 2'b00;
            axi_rdata   <= '0;
            rd_addr_reg <= '0;
        end else begin
            case (rd_state)
                RD_IDLE: begin
                    axi_rvalid <= 1'b0;
                    if (axi_arvalid) begin
                        axi_arready <= 1'b1;
                        rd_addr_reg <= axi_araddr;
                        rd_state    <= RD_DATA;
                    end
                end
                
                RD_DATA: begin
                    axi_arready <= 1'b0;
                    
                    // Decode address and read register
                    case (rd_addr_reg[7:0])
                        8'h00: axi_rdata <= ctrl_reg;
                        8'h04: axi_rdata <= {30'b0, status_done, status_busy};
                        8'h08: axi_rdata <= weight_reg;
                        8'h10: axi_rdata <= cycle_count;
                        8'h14: axi_rdata <= compute_cycles;
                        8'h18: axi_rdata <= data_count;
                        8'h1C: axi_rdata <= VERSION;
                        default: axi_rdata <= 32'hDEAD_BEEF;
                    endcase
                    
                    axi_rvalid <= 1'b1;
                    axi_rresp  <= 2'b00; // OKAY
                    
                    if (axi_rready) begin
                        rd_state <= RD_IDLE;
                    end
                end
                
                default: rd_state <= RD_IDLE;
            endcase
        end
    end

    // =========================================================
    // Control signal extraction
    // =========================================================
    
    // Start is a pulse (one-shot)
    logic ctrl_start_prev;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            ctrl_start_prev <= 1'b0;
        else
            ctrl_start_prev <= ctrl_reg[0];
    end
    assign ctrl_start = ctrl_reg[0] & ~ctrl_start_prev;
    assign ctrl_reset = ctrl_reg[1];
    assign ctrl_mode  = ctrl_reg[15:8];
    
    // Weight interface
    assign weight_data  = weight_reg[7:0];
    assign weight_valid = weight_ctrl_reg[0];

endmodule
