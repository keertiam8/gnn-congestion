read_lef sky130_fd_sc_hd.tlef
read_lef sky130_fd_sc_hd_merged.lef
read_liberty sky130_fd_sc_hd__tt_025C_1v80.lib

read_verilog pwm_gen_netlist.v
link_design pwm_gen

initialize_floorplan -die_area {0 0 60 60} \
                     -core_area {5 5 55 55} \
                     -site unithd

make_tracks

place_pins -hor_layer met3 -ver_layer met2

global_placement -density 0.6

detailed_placement

global_route
