# File: scripts/extract_gcd_data.tcl
# Extract GCD netlist, positions, and metrics from OpenROAD .odb

proc extract_gcd_data {odb_file output_dir} {
    # Read the design
    read_odb $odb_file
    
    # Get design name
    set design [get_name [get_design]]
    puts "Design: $design"
    
    # Extract instances (cells)
    set instances [get_insts]
    puts "Total instances: [llength $instances]"
    
    # Dump instance positions to CSV
    set f [open "$output_dir/instances.csv" w]
    puts $f "instance_name,x_min,y_min,x_max,y_max"
    foreach inst $instances {
        set bbox [get_property [get_bounding_box $inst]]
        set name [get_name $inst]
        puts $f "$name,[lindex $bbox 0],[lindex $bbox 1],[lindex $bbox 2],[lindex $bbox 3]"
    }
    close $f
    puts "Wrote instances.csv"
    
    # Extract nets
    set nets [get_nets]
    puts "Total nets: [llength $nets]"
    
    # Dump connectivity (net -> pins -> instances)
    set f [open "$output_dir/connectivity.csv" w]
    puts $f "net_name,instance_name,pin_name"
    foreach net $nets {
        set net_name [get_name $net]
        set pins [get_pins -of_objects $net]
        foreach pin $pins {
            set pin_name [get_name $pin]
            set inst [get_property cell_name [get_insts -of_objects $pin]]
            if {$inst ne ""} {
                puts $f "$net_name,$inst,$pin_name"
            }
        }
    }
    close $f
    puts "Wrote connectivity.csv"
    
    # Report design metrics
    report_design_metrics > "$output_dir/design_metrics.txt"
    
    puts "Extraction complete. Output in: $output_dir"
}

# Main
if {[llength $argv] < 2} {
    puts "Usage: extract_gcd_data <odb_file> <output_dir>"
    exit 1
}

set odb_file [lindex $argv 0]
set output_dir [lindex $argv 1]

extract_gcd_data $odb_file $output_dir