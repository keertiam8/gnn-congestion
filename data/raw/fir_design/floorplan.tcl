read_lef sky130_fd_sc_hd.tlef
read_lef sky130_fd_sc_hd_merged.lef
read_liberty sky130_fd_sc_hd__tt_025C_1v80.lib
read_verilog fir_filter_netlist.v
link_design fir_filter
initialize_floorplan -die_area {0 0 130 130} \
                     -core_area {5 5 125 125} \
                     -site unithd
make_tracks
place_pins -hor_layer met3 -ver_layer met2
insert_tiecells sky130_fd_sc_hd__conb_1/LO
insert_tiecells sky130_fd_sc_hd__conb_1/HI
global_placement -density 0.6
detailed_placement
global_route -guide_file results/fir_congestion.guide -congestion_report_file results/fir_congestion.rpt
write_def results/fir_placed.def
write_db results/fir_placed.odb
detailed_route -output_drc results/fir_drc.rpt -output_maze results/fir_maze.log -verbose 1
write_def results/fir_routed.def
write_db results/fir_routed.odb
exit
