# ============================================
# UART OpenROAD Floorplan + Placement + Routing
# ============================================

# Read technology files
read_lef sky130_fd_sc_hd.tlef
read_lef sky130_fd_sc_hd_merged.lef
read_liberty sky130_fd_sc_hd__tt_025C_1v80.lib

# Read synthesized UART netlist
read_verilog uart_netlist.v
link_design uart

# --------------------------------------------
# Floorplan
# --------------------------------------------
# Increased from 60x60 to 90x90 because
# UART utilization was >100% at 60x60.
initialize_floorplan -die_area {0 0 90 90} \
                     -core_area {5 5 85 85} \
                     -site unithd

# --------------------------------------------
# Routing tracks
# --------------------------------------------
make_tracks

# --------------------------------------------
# Place I/O pins
# --------------------------------------------
place_pins -hor_layer met3 -ver_layer met2

# --------------------------------------------
# Global placement
# --------------------------------------------
global_placement -density 0.6

# --------------------------------------------
# Detailed placement
# --------------------------------------------
detailed_placement

# --------------------------------------------
# Global routing
# --------------------------------------------
global_route

# --------------------------------------------
# Save results
# --------------------------------------------
write_db results/uart_routed.odb
write_def results/uart_routed.def
write_guides results/uart_congestion.guide
