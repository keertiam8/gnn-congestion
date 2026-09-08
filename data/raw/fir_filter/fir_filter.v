module fir_filter (
    input  wire        clk,
    input  wire        rst_n,
    input  wire         sample_valid,
    input  wire signed [7:0]  sample_in,
    output reg  signed [17:0] result,
    output reg          result_valid
);
    localparam signed [7:0] C0 = 8'sd20;
    localparam signed [7:0] C1 = 8'sd40;
    localparam signed [7:0] C2 = 8'sd40;
    localparam signed [7:0] C3 = 8'sd20;

    reg signed [7:0] x0, x1, x2, x3;
    reg signed [17:0] p0, p1, p2, p3;
    reg signed [17:0] sum_stage1_a, sum_stage1_b;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x0 <= 8'sd0; x1 <= 8'sd0; x2 <= 8'sd0; x3 <= 8'sd0;
            p0 <= 18'sd0; p1 <= 18'sd0; p2 <= 18'sd0; p3 <= 18'sd0;
            sum_stage1_a <= 18'sd0;
            sum_stage1_b <= 18'sd0;
            result       <= 18'sd0;
            result_valid <= 1'b0;
        end else if (sample_valid) begin
            x3 <= x2; x2 <= x1; x1 <= x0; x0 <= sample_in;
            p0 <= x0 * C0;
            p1 <= x1 * C1;
            p2 <= x2 * C2;
            p3 <= x3 * C3;
            sum_stage1_a <= p0 + p1;
            sum_stage1_b <= p2 + p3;
            result       <= sum_stage1_a + sum_stage1_b;
            result_valid <= 1'b1;
        end else begin
            result_valid <= 1'b0;
        end
    end
endmodule
