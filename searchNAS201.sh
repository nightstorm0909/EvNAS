#!/bin/bash
# bash ./searchNAS201.sh cifar10 0 outputs

echo script name: $0
echo $# arguments

dataset=$1
gpu=$2
channel=16
num_cells=5
max_nodes=4
output_dir=$3
epochs=50

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="../data"
else
  data_path="../data/ImageNet16"
fi
api_path="../NAS_Bench_201/NAS-Bench-201-v1_1-096897.pth"
config_path="configs/EvNAS.config"

python train_searchNAS201.py --cutout --gpu ${gpu} --max_nodes ${max_nodes} --init_channel ${channel} --num_cells ${num_cells} \
                                 --dataset ${dataset} --data ${data_path} --dir ${output_dir} --seed 18 --epochs ${epochs} --pop_size 50 --tsize 10 \
                                 --api_path ${api_path} --config_path ${config_path} --track_running_stats

python train_searchNAS201.py --cutout --gpu ${gpu} --max_nodes ${max_nodes} --init_channel ${channel} --num_cells ${num_cells} \
                                 --dataset ${dataset} --data ${data_path} --dir ${output_dir} --seed 19 --epochs ${epochs} --pop_size 50 --tsize 10 \
                                 --api_path ${api_path} --config_path ${config_path} --track_running_stats

python train_searchNAS201.py --cutout --gpu ${gpu} --max_nodes ${max_nodes} --init_channel ${channel} --num_cells ${num_cells} \
                                 --dataset ${dataset} --data ${data_path} --dir ${output_dir} --seed 20 --epochs ${epochs} --pop_size 50 --tsize 10 \
                                 --api_path ${api_path} --config_path ${config_path} --track_running_stats
