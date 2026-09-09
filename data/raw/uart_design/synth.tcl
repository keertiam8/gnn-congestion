# Read and elaborate
read_verilog uart.v
hierarchy -check -top uart

# Generic synthesis
synth -top uart

# Map flip-flops to SKY130 library cells (THIS is what kills $_DFF_PP0_)
dfflibmap -liberty sky130_fd_sc_hd__tt_025C_1v80.lib

# Technology mapping with ABC
abc -liberty sky130_fd_sc_hd__tt_025C_1v80.lib

# Remove unused cells/wires
opt_clean -purge

# Write netlist (no -simplenet, that flag doesn't exist)
write_verilog -noattr -noexpr -nohex uart_netlist.v
