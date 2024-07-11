#!/bin/bash

set -x

curr_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
root_dir=$(dirname $curr_dir)

worker_num=20
for ((i=0; i<${worker_num}; i++))
do
    python ${root_dir}/img_fetch_shm.py ${worker_num} ${i} > ${root_dir}/log/shm_4th_${i}.log 2>&1
done 

