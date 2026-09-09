module uart #(
    parameter CLKS_PER_BIT = 16
)(
    input  wire       clk,
    input  wire       rst,

    // UART transmitter
    input  wire       tx_start,
    input  wire [7:0] tx_data,
    output reg        tx,
    output reg        tx_busy,

    // UART receiver
    input  wire       rx,
    output reg [7:0]  rx_data,
    output reg        rx_valid
);

    // =========================================================
    // TX
    // =========================================================

    localparam TX_IDLE  = 3'd0;
    localparam TX_START = 3'd1;
    localparam TX_DATA  = 3'd2;
    localparam TX_STOP  = 3'd3;

    reg [2:0] tx_state;
    reg [15:0] tx_clk_count;
    reg [2:0] tx_bit_count;
    reg [7:0] tx_shift_reg;

    always @(posedge clk) begin
        if (rst) begin
            tx_state     <= TX_IDLE;
            tx_clk_count <= 16'd0;
            tx_bit_count <= 3'd0;
            tx_shift_reg <= 8'd0;
            tx           <= 1'b1;
            tx_busy      <= 1'b0;
        end
        else begin
            case (tx_state)

                TX_IDLE: begin
                    tx          <= 1'b1;
                    tx_busy     <= 1'b0;
                    tx_clk_count <= 16'd0;
                    tx_bit_count <= 3'd0;

                    if (tx_start) begin
                        tx_shift_reg <= tx_data;
                        tx_busy      <= 1'b1;
                        tx_state     <= TX_START;
                    end
                end

                TX_START: begin
                    tx <= 1'b0;

                    if (tx_clk_count == CLKS_PER_BIT-1) begin
                        tx_clk_count <= 16'd0;
                        tx_state     <= TX_DATA;
                    end
                    else begin
                        tx_clk_count <= tx_clk_count + 1'b1;
                    end
                end

                TX_DATA: begin
                    tx <= tx_shift_reg[tx_bit_count];

                    if (tx_clk_count == CLKS_PER_BIT-1) begin
                        tx_clk_count <= 16'd0;

                        if (tx_bit_count == 3'd7) begin
                            tx_bit_count <= 3'd0;
                            tx_state     <= TX_STOP;
                        end
                        else begin
                            tx_bit_count <= tx_bit_count + 1'b1;
                        end
                    end
                    else begin
                        tx_clk_count <= tx_clk_count + 1'b1;
                    end
                end

                TX_STOP: begin
                    tx <= 1'b1;

                    if (tx_clk_count == CLKS_PER_BIT-1) begin
                        tx_clk_count <= 16'd0;
                        tx_state     <= TX_IDLE;
                        tx_busy      <= 1'b0;
                    end
                    else begin
                        tx_clk_count <= tx_clk_count + 1'b1;
                    end
                end

                default: begin
                    tx_state <= TX_IDLE;
                    tx       <= 1'b1;
                end

            endcase
        end
    end


    // =========================================================
    // RX
    // =========================================================

    localparam RX_IDLE  = 3'd0;
    localparam RX_START = 3'd1;
    localparam RX_DATA  = 3'd2;
    localparam RX_STOP  = 3'd3;

    reg [2:0] rx_state;
    reg [15:0] rx_clk_count;
    reg [2:0] rx_bit_count;
    reg [7:0] rx_shift_reg;

    always @(posedge clk) begin
        if (rst) begin
            rx_state     <= RX_IDLE;
            rx_clk_count <= 16'd0;
            rx_bit_count <= 3'd0;
            rx_shift_reg <= 8'd0;
            rx_data      <= 8'd0;
            rx_valid     <= 1'b0;
        end
        else begin

            // Default: rx_valid is a one-cycle pulse
            rx_valid <= 1'b0;

            case (rx_state)

                RX_IDLE: begin
                    rx_clk_count <= 16'd0;
                    rx_bit_count <= 3'd0;

                    if (rx == 1'b0) begin
                        rx_state <= RX_START;
                    end
                end

                RX_START: begin

                    // Sample in the middle of start bit
                    if (rx_clk_count == (CLKS_PER_BIT/2)-1) begin

                        if (rx == 1'b0) begin
                            rx_clk_count <= 16'd0;
                            rx_state     <= RX_DATA;
                        end
                        else begin
                            rx_state <= RX_IDLE;
                        end

                    end
                    else begin
                        rx_clk_count <= rx_clk_count + 1'b1;
                    end
                end

                RX_DATA: begin

                    if (rx_clk_count == CLKS_PER_BIT-1) begin
                        rx_clk_count <= 16'd0;

                        rx_shift_reg[rx_bit_count] <= rx;

                        if (rx_bit_count == 3'd7) begin
                            rx_bit_count <= 3'd0;
                            rx_state     <= RX_STOP;
                        end
                        else begin
                            rx_bit_count <= rx_bit_count + 1'b1;
                        end

                    end
                    else begin
                        rx_clk_count <= rx_clk_count + 1'b1;
                    end
                end

                RX_STOP: begin

                    if (rx_clk_count == CLKS_PER_BIT-1) begin
                        rx_clk_count <= 16'd0;

                        rx_data  <= rx_shift_reg;
                        rx_valid <= 1'b1;

                        rx_state <= RX_IDLE;
                    end
                    else begin
                        rx_clk_count <= rx_clk_count + 1'b1;
                    end
                end

                default: begin
                    rx_state <= RX_IDLE;
                end

            endcase
        end
    end

endmodule
