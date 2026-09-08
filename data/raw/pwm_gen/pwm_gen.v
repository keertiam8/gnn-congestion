module pwm_gen(
    input wire clk,
    input wire rst,
    input wire [7:0] duty_cycle,
    output reg pwm_out
);

    // 8 bit internal counter
    reg [7:0] count;

    // 1. Sequential block for the counter logic
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            count <= 8'b0;
        end else begin
            count <= count + 1'b1;
        end
    end

    // 2. Sequential block for the comparator logic
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            pwm_out <= 1'b0;
        end else if (count < duty_cycle) begin
            pwm_out <= 1'b1;
        end else begin
            pwm_out <= 1'b0;
        end
    end

endmodule
