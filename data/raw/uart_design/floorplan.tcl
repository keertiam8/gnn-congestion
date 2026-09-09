read_lef sky130_fd_sc_hd.tlef
read_lef sky130_fd_sc_hd_merged.lef
read_liberty sky130_fd_sc_hd__tt_025C_1v80.lib

read_verilog uart_netlist.v
link_design uart

initialize_floorplan -die_area {0 0 100 100} \
                      -core_area {5 5 95 95} \
                      -site unithd

make_tracks

place_pins -hor_layer met3 -ver_layer met2

global_placement -density 0.6

detailed_placement

# Save PLACED results (before routing)
write_db results/uart_placed.odb
write_def results/uart_placed.def

global_route

# Detailed routing (needed for real DRC report)
detailed_route -output_drc results/uart_drc.rpt

# Save ROUTED results
write_db results/uart_routed.odb
write_def results/uart_routed.def
write_guides results/uart_congestion.guide
