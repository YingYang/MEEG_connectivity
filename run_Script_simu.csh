#!/bin/tcsh

set L_list_option = 1

foreach i (`seq 1 1 5`)
    python Script_simu.py $i ${L_list_option}
end
