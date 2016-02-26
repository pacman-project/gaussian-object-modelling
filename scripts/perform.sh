#!/bin/bash

declare -a objects=("bowlA" "bowlB" "containerA" "containerB" "jug" "kettle" "kitchenUtensilB" "mugD" "pot")

for ((i=0; i<=2; i++)); do
    rosparam set /gaussian_process/global_goal 0.05
    rosparam set /gaussian_process/touch_type $i
    rosparam set /gaussian_process/simulate_touch true
    for obj in "${objects[@]}"
    do
        rosservice call /gaussian_process/start_process "obj_pcd: '${PWD}/../resources/${obj}.pcd'"
        rosservice call /gaussian_process/get_next_best_path "{var_desired: {data: 0.4}}"
        file=${PWD}/../results/${obj}_${i}.pcd
        while [ ! -f ${file} ]
        do
            sleep 120
        done
        echo "Generated ${obj} with type ${i}"
    done
done
echo "=============================================="
echo "=========================== > Done all tests <"
echo "=============================================="

