#!/bin/bash

if [ -e gpu_log.csv ]; then
    rm -f gpu_log.csv
fi

touch gpu_log.csv

while true; do
    # 获取 CSV 标题行
    csv_header=$(head -n 1 gpu_log.csv)

    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv >> gpu_log.csv

    # 删除重复行
    sort -u -o gpu_log.csv gpu_log.csv

    # 将标题行删除
    sed -i '1d' gpu_log.csv

    # 将标题行追加到日志文件顶部
    echo "$csv_header" | cat - gpu_log.csv > /tmp/gpu_log.csv && mv /tmp/gpu_log.csv gpu_log.csv

    sleep 1
done
