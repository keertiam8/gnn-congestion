# Extract GCD netlist, positions from OpenROAD .odb
# Run with: openroad scripts/extract_gcd_data.tcl

set output_dir "/foss/designs/OpenROAD-flow-scripts/flow/results/sky130hd/gcd/base/extracted_data"
file mkdir $output_dir

# Read the placed design
read_db /foss/designs/OpenROAD-flow-scripts/flow/results/sky130hd/gcd/base/3_place.odb

# Get design
set db [ord::get_db]
set chip [$db getChip]
set block [$chip getBlock]

# Get instances
set instances [$block getInsts]
puts "Total instances: [llength $instances]"

# Write instances CSV
set f [open "$output_dir/instances.csv" w]
puts $f "instance_name,x_min,y_min,x_max,y_max"
foreach inst $instances {
    set name [$inst getName]
    set bbox [$inst getBBox]
    set x_min [$bbox xMin]
    set y_min [$bbox yMin]
    set x_max [$bbox xMax]
    set y_max [$bbox yMax]
    puts $f "$name,$x_min,$y_min,$x_max,$y_max"
}
close $f
puts "Wrote instances.csv"

# Get nets
set nets [$block getNets]
puts "Total nets: [llength $nets]"

# Write connectivity CSV
set f [open "$output_dir/connectivity.csv" w]
puts $f "net_name,instance_name,pin_name"
foreach net $nets {
    set net_name [$net getName]
    set iterms [$net getITerms]
    foreach iterm $iterms {
        set inst [$iterm getInst]
        if {$inst ne ""} {
            set inst_name [$inst getName]
            set mterm [$iterm getMTerm]
            set pin_name [$mterm getName]
            puts $f "$net_name,$inst_name,$pin_name"
        }
    }
}
close $f
puts "Wrote connectivity.csv"

puts "Extraction complete!"
exit